"""Freeze and fetch public TCGA patient-level inputs. Standard library only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import ssl
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://api.gdc.cancer.gov"
KINDS = ("he", "rna", "clinical")
TYPES = ("Slide Image", "Gene Expression Quantification", "Clinical Supplement")
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
PROJECT = re.compile(r"TCGA-[A-Z0-9]+\Z")
PATIENT = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\Z")
CASE_FIELDS = "case_id,submitter_id,project.project_id"
FILE_FIELDS = ("file_id,file_name,file_size,md5sum,state,access,data_type,data_format,experimental_strategy,"
               "analysis.workflow_type,cases.case_id,cases.submitter_id,cases.project.project_id,"
               "cases.samples.submitter_id,cases.samples.sample_type")
FIELDS = ["sample_id", "he_path", "project_id", "case_id", "patient_id", "pairing_scope"] + [
    f"{k}_{s}" for k in KINDS for s in ("path", "file_id", "file_name", "bytes", "md5", "url", "sample_barcodes", "eligible_file_count")
    if (k,s) != ("he", "path")]


def https_open(request, *, timeout):
    # Use platform roots on a cloud host. An explicit CA bundle also works in
    # environments whose platform root store is unavailable; TLS stays verified.
    cafile=os.environ.get('SSL_CERT_FILE')
    if cafile:return urlopen(request,timeout=timeout,context=ssl.create_default_context(cafile=cafile))
    return urlopen(request,timeout=timeout)


def jsonout(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""): h.update(chunk)
    return h.hexdigest()


def transient(e):
    return (e.code in (408, 429) or 500 <= e.code < 600) if isinstance(e, HTTPError) else isinstance(e,(URLError, TimeoutError, ConnectionError, http.client.IncompleteRead))


def metadata(endpoint, params):
    url = f"{API}/{endpoint}?" + urlencode(params)
    for attempt in range(5):
        try:
            with https_open(Request(url, headers={"Accept": "application/json"}), timeout=60) as res:
                return json.load(res)["data"]
        except (OSError, http.client.IncompleteRead) as e:
            if attempt == 4 or not transient(e): raise
            time.sleep(2 ** attempt)


def query(project, endpoint):
    clause = lambda field, values: {"op": "in", "content": {"field": field, "value": values}}
    if endpoint == "cases": return clause("project.project_id", [project]), CASE_FIELDS
    return {"op": "and", "content": [clause("cases.project.project_id", [project]), clause("access", ["open"]),
                                       clause("state", ["released"]), clause("data_type", list(TYPES))]}, FILE_FIELDS


def pages(project, endpoint, *, page_size=250, get=metadata):
    if not PROJECT.fullmatch(project) or endpoint not in ("cases", "files") or not 1 <= page_size <= 1000:
        raise ValueError("Invalid TCGA project, endpoint, or page size")
    filt, fields = query(project, endpoint); key = "case_id" if endpoint == "cases" else "file_id"
    offset, total, seen = 0, None, set()
    while total is None or offset < total:
        data = get(endpoint, dict(filters=json.dumps(filt), fields=fields, sort=f"{key}:asc", size=page_size,
                                  format="JSON", **{"from":offset}))
        pag, hits = data["pagination"], data["hits"]
        if total is None: total = int(pag["total"])
        if (total < 0 or int(pag["total"]) != total or ("from" in pag and int(pag["from"]) != offset)
                or int(pag.get("count",len(hits))) != len(hits) or offset+len(hits)>total or (offset<total and not hits)):
            raise ValueError("GDC pagination changed during retrieval; start a fresh snapshot")
        for hit in hits:
            uid = hit[key]
            if uid in seen: raise ValueError("Repeated GDC ID across pages; restart discovery")
            seen.add(uid); yield hit
        offset += len(hits)


def source(file):
    uid,name = file.get("file_id",""),file.get("file_name","")
    if (not UUID.fullmatch(uid) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",name)
            or not re.fullmatch(r"[a-f0-9]{32}",str(file.get("md5sum","")))
            or not isinstance(file.get("file_size"),int) or file["file_size"] < 1):
        return None
    return uid,name


def paired(project, cases, files):
    by_case, patients, rejected = {},set(),Counter(); opts=defaultdict(lambda:defaultdict(list))
    for case in cases:
        uid, sid = case["case_id"],case["submitter_id"]
        if (not UUID.fullmatch(uid) or not PATIENT.fullmatch(sid) or case["project"]["project_id"]!=project
                or uid in by_case or sid in patients): raise ValueError("Unexpected or duplicate GDC participant identity")
        by_case[uid]=sid; patients.add(sid)
    for f in files:
        cs=f.get("cases") or []
        if len(cs)!=1 or cs[0].get("case_id") not in by_case:
            rejected["not_one_matching_case"]+=1;continue
        c=cs[0]; cid=c["case_id"]
        if (c.get("submitter_id")!=by_case[cid] or (c.get("project") or {}).get("project_id")!=project):
            raise ValueError("File case/project identity differs from case inventory")
        if f.get("access")!="open" or f.get("state")!="released" or source(f) is None:
            rejected["not_verified_open_released_metadata"]+=1;continue
        dt, fmt = f.get("data_type"), f.get("data_format")
        if dt==TYPES[0] and f.get("experimental_strategy")=="Diagnostic Slide" and fmt=="SVS" and f['file_name'].lower().endswith('.svs'): k="he"
        elif dt==TYPES[1] and (f.get("analysis") or {}).get("workflow_type")=="STAR - Counts" and fmt=="TSV": k="rna"
        elif dt==TYPES[2] and fmt=="BCR XML" and f['file_name'].lower().endswith('.xml'): k="clinical"
        else: rejected["not_target_file_type"]+=1;continue
        sam=c.get("samples") or []
        if k!="clinical" and (not sam or any(s.get("sample_type")!="Primary Tumor" for s in sam)):
            rejected["missing_or_non_primary_sample"]+=1;continue
        barcodes=sorted({s["submitter_id"] for s in sam if s.get("submitter_id")});
        if k!="clinical" and (not barcodes or any(not b.startswith(by_case[cid]+"-") for b in barcodes)):
            rejected["missing_or_inconsistent_sample_barcode"]+=1;continue
        opts[cid][k].append((tuple(barcodes),f["file_name"],f["file_id"],f))
    rows=[]; excluded=[]
    for cid,sid in sorted(by_case.items(),key=lambda p:p[1]):
        missing=[k for k in KINDS if not opts[cid][k]]
        if missing:
            excluded.append(dict(project_id=project,case_id=cid,patient_id=sid,missing_kinds=";".join(missing)));continue
        row=dict(sample_id=sid,patient_id=sid,case_id=cid,project_id=project,pairing_scope="same_patient_not_spatially_registered")
        for k in KINDS:
            candidates=opts[cid][k]; bars,name,uid,f=min(candidates,key=lambda t:t[:3])
            fields=dict(file_id=uid,file_name=name,path=f"raw/{k}/{uid}/{name}",bytes=f["file_size"],md5=f["md5sum"],
                        url=f"{API}/data/{uid}",sample_barcodes=";".join(bars),eligible_file_count=len(candidates))
            row.update({f"{k}_{s}":v for s,v in fields.items()})
        rows.append(row)
    return rows,excluded,dict(rejected)


def csvout(p,cols,rows):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,lineterminator="\n");w.writeheader();w.writerows(rows)


def prepare(project, destination, *, page_size=250, get=metadata):
    dest=Path(destination).resolve()
    if dest.exists(): raise FileExistsError(f"Snapshot destination already exists: {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage=Path(tmp)/'snapshot';stage.mkdir()
        cases=list(pages(project,'cases',page_size=page_size,get=get))
        if not cases: raise ValueError(f"No cases in {project}; check the GDC project ID")
        files=list(pages(project,'files',page_size=page_size,get=get)); rows,excluded,rejected=paired(project,cases,files)
        for name,data in [('cases',cases),('files',files)]:jsonout(stage/f'gdc_{name}.json',data)
        csvout(stage/'manifest.csv',FIELDS,rows);csvout(stage/'excluded.csv',['project_id','case_id','patient_id','missing_kinds'],excluded)
        summary=dict(project_id=project,retrieved_at_utc=datetime.now(timezone.utc).isoformat(),api=API,
                     cases=len(cases),queried_open_files=len(files),matched=len(rows),excluded=len(excluded),rejected_file_counts=rejected,
                     queries={name:dict(filters=query(project,name)[0],fields=query(project,name)[1]) for name in ('cases','files')},
                     selection='Within each patient and file type: lexical sample barcode(s), filename, UUID; no score/outcome or file-size selection.',
                     expected_bytes={k:sum(r[k+'_bytes'] for r in rows) for k in KINDS},
                     sha256={p.name:digest(p,'sha256') for p in sorted(stage.iterdir())})
        jsonout(stage/'snapshot.json',summary);stage.rename(dest)
    return summary


@contextmanager
def file_lock(path):
    with path.open('a+b') as lock:
        if os.name=='nt':
            import msvcrt
            if lock.seek(0,2)==0:lock.write(b'\0');lock.flush()
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def fetch_row(root,row,kind,*,opener=https_open,attempts=5):
    uid,name=row[kind+'_file_id'],row[kind+'_file_name'];size=int(row[kind+'_bytes']);md5=row[kind+'_md5']
    if source(dict(file_id=uid,file_name=name,file_size=size,md5sum=md5)) is None:raise ValueError('Invalid file metadata')
    rel=f'raw/{kind}/{uid}/{name}';url=f'{API}/data/{uid}'
    if row[kind+'_path']!=rel or row[kind+'_url']!=url:raise ValueError('File path/URI does not match GDC UUID')
    root=Path(root).resolve();target=root/rel
    if not target.resolve().is_relative_to(root) or any(p.is_symlink() for p in (target,*target.parents) if p!=root.parent):raise ValueError('Unexpected output link')
    target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix(target.suffix+'.partial');lock=target.with_suffix(target.suffix+'.lock')
    if partial.is_symlink() or lock.is_symlink():raise ValueError('Unsafe transfer path')
    with file_lock(lock):
        if target.exists():
            if target.stat().st_size!=size or digest(target,'md5')!=md5: raise ValueError(f'Existing original mismatch; untouched: {target}')
            return 'reused_verified'
        for n in range(attempts):
            off=partial.stat().st_size if partial.exists() else 0
            if off>size:raise ValueError('Partial file exceeds expected size')
            try:
                if off<size:
                    headers={'Accept-Encoding':'identity'}
                    if off:headers['Range']=f'bytes={off}-'
                    with opener(Request(url,headers=headers),timeout=90) as res:
                        if res.status==206:
                            bounds=re.fullmatch(r'(?:bytes )?(\d+)-(\d+)/(\d+)',res.headers.get('Content-Range',''))
                            if not bounds or tuple(map(int,bounds.groups()))!=(off,size-1,size):raise ValueError('Wrong GDC Content-Range')
                        elif off or res.status!=200:raise ValueError('GDC did not honor requested range')
                        if res.headers.get('Content-Encoding','identity')!='identity':raise ValueError('Unexpected transfer encoding')
                        if res.headers.get('Content-Length') is not None and int(res.headers['Content-Length'])!=size-off:raise ValueError('Wrong Content-Length')
                        with partial.open('ab' if partial.exists() else 'xb') as writer:
                            for chunk in iter(lambda:res.read(1024**2),b''):
                                off+=len(chunk)
                                if off>size:raise ValueError('Response exceeds expected file size')
                                writer.write(chunk)
                    if off<size:raise http.client.IncompleteRead(b'')
                if digest(partial,'md5')!=md5:raise ValueError(f'MD5 mismatch; partial retained: {partial}')
                if os.name=='nt':partial.rename(target)
                else:os.link(partial,target);partial.unlink()
                return 'downloaded_verified'
            except (OSError,http.client.IncompleteRead) as e:
                if n+1==attempts or not transient(e):raise
                time.sleep(2**n)


def download(manifest, samples, kinds):
    p=Path(manifest).resolve();snap=json.loads((p.parent/'snapshot.json').read_text(encoding='utf-8'))
    if p.name!='manifest.csv' or snap['sha256'].get('manifest.csv')!=digest(p,'sha256'): raise ValueError('Frozen manifest digest mismatch')
    with p.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No matched patients in this snapshot')
    if not set(samples)<=set(r['sample_id'] for r in rows):raise ValueError('Unknown sample ID')
    for row in rows:
        if not samples or row['sample_id'] in samples:
            for k in (kinds or KINDS):print(row['sample_id'],k,fetch_row(p.parent,row,k),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);cmd=parser.add_subparsers(dest='command',required=True)
    prep=cmd.add_parser('prepare');prep.add_argument('--project',required=True);prep.add_argument('--out',required=True,type=Path);prep.add_argument('--page-size',type=int,default=250)
    dl=cmd.add_parser('download');dl.add_argument('--manifest',required=True,type=Path);dl.add_argument('--sample',action='append',default=[]);dl.add_argument('--kind',choices=KINDS,action='append',default=[])
    args=parser.parse_args()
    if args.command=='prepare':
        out=prepare(args.project.upper(),args.out,page_size=args.page_size)
        print(json.dumps({k:out[k] for k in ['project_id','cases','matched','excluded','expected_bytes']}))
    else:download(args.manifest,args.sample,args.kind)


if __name__=='__main__':main()
