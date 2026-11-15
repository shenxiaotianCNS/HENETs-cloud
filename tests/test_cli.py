import pytest
from henets.cli import clear_verified_cache, load_samples, select


def test_relative_manifest_and_disjoint_shards(tmp_path):
    path = tmp_path/"slides.csv"
    path.write_text("sample_id,he_path,cancer_type\nliver,test/l.svs,liver\nbreast,test/b.svs,breast\n")
    cases = load_samples(path)
    assert cases[0]["he_path"] == str((tmp_path/"test/l.svs").resolve())
    assert select(cases, None, "0/2")+select(cases, None, "1/2") == cases
    with pytest.raises(ValueError):
        select(cases, None, "2/2")
    path.write_text("sample_id,he_path\n../bad,l.svs\n")
    with pytest.raises(ValueError):
        load_samples(path)


def test_cleanup_validates_all_names_before_unlinking(tmp_path):
    blocks = tmp_path/"blocks"
    blocks.mkdir()
    good = blocks/"x0_y0_labels.png"
    good.touch()
    unexpected = blocks/"annotator-note.txt"
    unexpected.touch()
    with pytest.raises(ValueError):
        clear_verified_cache(tmp_path)
    assert good.exists() and unexpected.exists()
    unexpected.unlink()
    clear_verified_cache(tmp_path)
    assert not blocks.exists()
