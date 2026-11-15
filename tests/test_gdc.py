"""Synthetic metadata and a six-byte file: no real projects, network or slides."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

# Avoid importing the research repository's identically named local package.
import importlib.util
spec=importlib.util.spec_from_file_location('cloud_public_gdc',Path(__file__).resolve().parents[1]/'henets/gdc.py')
gdc=importlib.util.module_from_spec(spec);spec.loader.exec_module(gdc)


def case(n):return {'case_id':str(UUID(int=n)),'submitter_id':f'TCGA-ZZ-{n:04}','project':{'project_id':'TCGA-TEST'}}


def file(kind,ix,case):
    sid=case['submitter_id']; uid=str(UUID(int=10+ix)); name={'he':'case.svs','rna':'counts.tsv','clinical':'clinical.xml'}[kind]
    return dict(file_id=uid,file_name=name,file_size=6,md5sum=hashlib.md5(b'abcdef').hexdigest(),access='open',state='released',
                cases=[{**case,'samples':[dict(submitter_id=sid+'-01A',sample_type='Primary Tumor')]}],
                data_type=dict(zip(gdc.KINDS,gdc.TYPES))[kind],data_format={'he':'SVS','rna':'TSV','clinical':'BCR XML'}[kind],
                experimental_strategy='Diagnostic Slide' if kind=='he' else None,analysis={'workflow_type':'STAR - Counts'})


class Response(io.BytesIO):
    def __init__(self,body,status,headers):super().__init__(body);self.status=status;self.headers=headers


class TestGDC(unittest.TestCase):
    def test_pagination_matching_and_frozen_snapshot(self):
        cases=[case(1),case(2)];files=[file(k,j,cases[0]) for j,k in enumerate(gdc.KINDS)]+[file('he',7,cases[1])]
        extra=dict(file('he',8,cases[0]),file_name='zzz.svs');files.append(extra)
        files.append(dict(file('rna',9,cases[1]),access='controlled'))
        normal=file('rna',10,cases[1]);normal['cases'][0]['samples'][0]['sample_type']='Solid Tissue Normal';files.append(normal)
        multiple=file('rna',11,cases[1]);multiple['cases'].append(cases[0]);files.append(multiple)
        calls=[]
        def get(endpoint,params):
            data=cases if endpoint=='cases' else files;off=params['from'];a=data[off:off+params['size']];calls.append((endpoint,off))
            return dict(hits=a,pagination=dict(total=len(data),count=len(a),**{'from':off}))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cohort'; out=gdc.prepare('TCGA-TEST',root,page_size=2,get=get)
            self.assertEqual((out['cases'],out['matched'],out['excluded']),(2,1,1));self.assertIn(('files',6),calls)
            with (root/'manifest.csv').open() as fp:row=next(csv.DictReader(fp))
            self.assertEqual(row['he_eligible_file_count'],'2');self.assertEqual(row['he_file_name'],'case.svs');self.assertFalse(Path(row['he_path']).is_absolute())
            with (root/'excluded.csv').open() as fp:self.assertEqual(next(csv.DictReader(fp))['missing_kinds'],'rna;clinical')
            self.assertEqual(out['sha256']['manifest.csv'],gdc.digest(root/'manifest.csv','sha256'))
            with self.assertRaises(FileExistsError):gdc.prepare('TCGA-TEST',root,page_size=2,get=get)
            (root/'manifest.csv').write_text('overwritten')
            with self.assertRaisesRegex(ValueError,'digest'):gdc.download(root/'manifest.csv',[],[])

    def test_changed_pagination_stops(self):
        def get(_,p):return {'hits':[case(1)],'pagination':{'total':2,'count':1,'from':p['from']}}
        with self.assertRaisesRegex(ValueError,'Repeated'):list(gdc.pages('TCGA-TEST','cases',page_size=1,get=get))

    def test_mismatched_patient_rejected(self):
        c=case(1);f=file('he',2,c);f['cases'][0]['submitter_id']='TCGA-ZZ-AAAA'
        with self.assertRaisesRegex(ValueError,'identity'):gdc.paired('TCGA-TEST',[c],[f])

    def test_download_resume_reuse_md5_and_hard_failure(self):
        c=case(1); rows,*_=gdc.paired('TCGA-TEST',[c],[file(k,j,c) for j,k in enumerate(gdc.KINDS)]);row=rows[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=root/row['he_path'];result.parent.mkdir(parents=True);partial=result.with_suffix('.svs.partial');partial.write_bytes(b'abc')
            with self.assertRaisesRegex(ValueError,'range'):
                gdc.fetch_row(root,row,'he',opener=lambda *a,**k:Response(b'abcdef',200,{}))
            self.assertEqual(partial.read_bytes(),b'abc')
            def opened(req,**kw):
                self.assertEqual(req.get_header('Range'),'bytes=3-');return Response(b'def',206,{'Content-Range':'bytes 3-5/6'})
            self.assertEqual(gdc.fetch_row(root,row,'he',opener=opened),'downloaded_verified');self.assertEqual(result.read_bytes(),b'abcdef')
            self.assertEqual(gdc.fetch_row(root,row,'he',opener=lambda *a,**k:self.fail('redownload')),'reused_verified')
            result.write_bytes(b'tamper');
            with self.assertRaisesRegex(ValueError,'untouched'):gdc.fetch_row(root,row,'he')
            self.assertEqual(result.read_bytes(),b'tamper')
            with self.assertRaisesRegex(ValueError,'path/URI'):gdc.fetch_row(root,dict(row,he_path='../escape'),'he')

    def test_occupied_lock_prevents_partial_write_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock=Path(tmp)/'target.lock'
            with gdc.file_lock(lock):
                with self.assertRaises(OSError):
                    with gdc.file_lock(lock):self.fail('duplicate lock owner')
            with gdc.file_lock(lock):pass


if __name__=='__main__':unittest.main()
