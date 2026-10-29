"""Independent X16-X20: actual archives, release state, Git CAS, SAE and event replay."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from harness.contracts.wp20_reference_b import dot, sha, serial


def x16(p, root):
    if set(p)!={"files","corrupt"}:
        raise ValueError("schema")
    root=Path(root)
    if not isinstance(p["files"],dict):
        raise ValueError("files")
    for path in p["files"]:
        if not isinstance(path,str) or not path or path.startswith(("/", "~")) or "\\" in path or ":" in path or any(v in ("..",".","") for v in path.split("/")):
            raise ValueError("unsafe archive member")
    path=root/"bundle.tar"
    with tarfile.open(path,"w") as tar:
        for name,text in sorted(p["files"].items()):
            data=text.encode("utf-8")
            info=tarfile.TarInfo(name)
            info.size=len(data)
            tar.addfile(info, io.BytesIO(data))
    if p["corrupt"]:
        raise ValueError("simulated archive failure")
    target=root/"extract"
    target.mkdir()
    unpacked={}
    with tarfile.open(path,"r") as tar:
        for info in tar.getmembers():
            if info.name not in p["files"] or not info.isfile():
                raise ValueError("unsafe tar")
            data=tar.extractfile(info).read()
            dest=(target/info.name).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError("archive escape")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            unpacked[info.name]=hashlib.sha256(data).hexdigest()
    return {"files":sorted(unpacked),"content_sha256":unpacked,
            "archive_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "verified":True}


def x17(p, root):
    if set(p)!={"source_sha","checks","required","reviewer"}:
        raise ValueError("schema")
    required=p["required"]
    if len(set(required))!=len(required) or not required or not isinstance(p["source_sha"],str):
        raise ValueError("required")
    by_id={r["id"]:r for r in p["checks"]}
    if len(by_id)!=len(p["checks"]) or any(x not in by_id for x in required):
        raise ValueError("missing check")
    if any(by_id[x]["source_sha"]!=p["source_sha"] or by_id[x]["state"]!="success" for x in required):
        raise ValueError("incomplete release gates")
    if p["reviewer"]!="trusted-human":
        raise ValueError("no human approval")
    return {"source_sha":p["source_sha"],"required":sorted(required),
            "release_ready":True,"published":False}


def _git(root,*args):
    r=subprocess.run(["git","-C",str(root),*args],text=True,capture_output=True,check=False,
                     timeout=5,env={"PATH":"/usr/bin:/bin","HOME":str(root),"GIT_CONFIG_NOSYSTEM":"1"})
    if r.returncode:
        raise ValueError("git failed")
    return r.stdout.strip()


def x18(p, root):
    if set(p)!={"base","a","b","rebase_b"}:
        raise ValueError("schema")
    root=Path(root)
    _git(root,"init")
    _git(root,"config","user.email","exam@example.invalid")
    _git(root,"config","user.name","Synthetic Exam")
    target=root/"module.py"
    target.write_text(p["base"],encoding="utf-8")
    _git(root,"add","module.py")
    _git(root,"commit","-m","base")
    base=_git(root,"rev-parse","HEAD")
    if not isinstance(p["a"],str) or not isinstance(p["b"],str):
        raise ValueError("edits")
    target.write_text(p["a"],encoding="utf-8")
    _git(root,"add","module.py")
    _git(root,"commit","-m","agent-a")
    head_a=_git(root,"rev-parse","HEAD")
    if head_a==base:
        raise ValueError("no change from A")
    if p["rebase_b"]!=head_a:
        raise ValueError("stale B expected SHA")
    target.write_text(p["b"],encoding="utf-8")
    _git(root,"add","module.py")
    _git(root,"commit","-m","agent-b-after-rebase")
    head_b=_git(root,"rev-parse","HEAD")
    return {"base_sha":base,"head_a_sha":head_a,"head_b_sha":head_b,
            "commits":3,"source":target.read_text(encoding="utf-8"),
            "history":["base","agent-a","agent-b-after-rebase"]}


def x19(p, root):
    if set(p)!={"samples","encoder","decoder","ablate"}:
        raise ValueError("schema")
    if not p["samples"]:
        raise ValueError("no samples")
    enc,dec=p["encoder"],p["decoder"]
    if len(enc)!=len(p["samples"][0]["vector"]) or len(dec)!=len(enc[0]):
        raise ValueError("shape")
    result=[];diffs=[]
    for sample in sorted(p["samples"],key=lambda s:s["id"]):
        x=sample["vector"]
        dense=[max(0,dot(x,[enc[i][j] for i in range(len(enc))])) for j in range(len(enc[0]))]
        idx=p["ablate"]
        if type(idx) is not int or idx<0 or idx>=len(dense):
            raise ValueError("feature")
        before=sum(dot(dense,[dec[j][i] for j in range(len(dec))]) for i in range(len(x)))
        changed=dense[:]; changed[idx]=0
        after=sum(dot(changed,[dec[j][i] for j in range(len(dec))]) for i in range(len(x)))
        result.append({"id":sample["id"],"features":dense,"prediction_before":before,"prediction_after":after})
        diffs.append(after-before)
    return {"samples":result,"ablation_feature":p["ablate"],
            "mean_prediction_delta":sum(diffs)/len(diffs),
            "coverage":{"evaluated":len(result),"total":len(p["samples"])}}


def x20(p, root):
    if set(p)!={"initial","events"}:
        raise ValueError("schema")
    state=dict(p["initial"])
    if set(state)!={"revision","source_sha","owner","status"}:
        raise ValueError("initial")
    history=[];failures=[]
    for ev in p["events"]:
        before=dict(state)
        kind=ev["op"]
        if kind=="handoff":
            if ev["from"]!=state["owner"] or not ev["to"]:
                raise ValueError("handoff")
            state["owner"]=ev["to"];state["revision"]+=1
        elif kind=="change":
            if ev["actor"]!="trusted-human" or ev["expected"]!=state["revision"]:
                raise ValueError("unauthorized change")
            state["source_sha"]=ev["source_sha"];state["revision"]+=1
        elif kind=="fail":
            failures.append({"reason":ev["reason"],"revision":state["revision"]})
            state["status"]="partial"
        elif kind=="resume":
            if ev["revision"]!=state["revision"] or ev["owner"]!=state["owner"]:
                raise ValueError("stale recovery")
            state["status"]="active"
        elif kind=="review":
            if ev["actor"]!="trusted-human" or ev["source_sha"]!=state["source_sha"]:
                raise ValueError("approval")
            state["status"]="reviewed"
        else:
            raise ValueError("event")
        history.append({"op":kind,"before":before,"after":dict(state)})
    return {"final":state,"events":history,"failures":failures,
            "human_approved":state["status"]=="reviewed","count":len(history)}


CASES={"X16":x16,"X17":x17,"X18":x18,"X19":x19,"X20":x20}
