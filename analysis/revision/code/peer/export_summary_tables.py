"""Export manuscript sample tables and a comparison with the frozen peer outputs."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import pandas as pd
from config import DATA, REGIONS


def main():
    root=Path(__file__).resolve().parents[2];folder=root/"tables/peer"
    main_counts=pd.read_csv(folder/"sample_counts.csv")
    own=pd.read_csv(folder/"own_tendency_figure_source.csv")
    labels={"discovery":"发现样本","holdout":"主要玩家留出","strict":"统一严格留出"}
    sources={"discovery":(126000,2240354),"holdout":(31821,854485),"strict":(23193,548256)}
    rows=[];changes=[]
    for sample,label in labels.items():
        players,matches=sources[sample]
        rows.append({"样本":label,"范围":"来源清单","焦点玩家数":players,"唯一比赛数":matches,"队友记录数":None,"对手记录数":None,"说明":"来源清单；各分析按其有效记录规则筛选。"})
        g=main_counts[main_counts["sample"].eq(sample)]
        rows.append({"样本":label,"范围":"整局同伴关联主分析","焦点玩家数":int(g.n_focal.sum()),"唯一比赛数":int(g.n_match.sum()),"队友记录数":int(g.n_teammate.sum()),"对手记录数":int(g.n_opponent.sum()),"说明":"每名焦点玩家至少10场习惯比赛；按名单区域分组；剔除非标准位置及重复同队关系。"})
        g=own[own["sample"].eq(sample)&own.region.isin(REGIONS)&own.relation.eq("teammate")&own.outcome.eq("control_wards")&own.term.eq("recipient_own_habit")]
        rows.append({"样本":label,"范围":"自身与伙伴倾向联合模型","焦点玩家数":int(g.n_focal.sum()),"唯一比赛数":int(g.n_match.sum()),"队友记录数":int(g.n_rows.sum()),"对手记录数":None,"说明":"仅接收者亦有独立于该场比赛的倾向估计；表列队友关系模型，对应图4a。"})
        d=pd.read_csv(folder/sample/"dynamics.csv")
        g=d[d.region.isin(REGIONS)&d.exposure.eq("control_wards")&d.outcome.eq("control_wards_m01")&d.method.eq("oriv")&d.relation.eq("teammate")]
        rows.append({"样本":label,"范围":"前10分钟轨迹","焦点玩家数":int(g.n_focal.sum()),"唯一比赛数":int(g.n_match.sum()),"队友记录数":int(g.n_teammate.sum()),"对手记录数":int(g.n_opponent.sum()),"说明":"有效分钟时间线；分钟记录沿用真实比赛服务器筛选，单位为每个一分钟区间的计数。"})
        previous="holdout_strict" if sample=="strict" else sample
        before=pd.read_csv(DATA.parent/"tables/study2"/previous/"main_effects.csv")
        after=pd.read_csv(folder/sample/"main_effects.csv")
        def select(frame):return frame[frame.method.eq("oriv")&frame.region.eq("pooled")&frame.exposure.eq(frame.outcome)]
        keys=["exposure","outcome","relation"];columns=keys+["b","b_std","se_std","tost_equivalent"]
        comparison=select(before)[columns].merge(select(after)[columns],on=keys,suffixes=("_previous","_corrected"))
        comparison["sample"]=sample;changes.append(comparison)
    pd.DataFrame(rows).to_csv(folder/"sample_range_for_word.csv",index=False,encoding="utf-8-sig")
    pd.concat(changes).to_csv(folder/"correction_comparison.csv",index=False)
    manifest={"code":{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/"code/peer").glob("*.py")},"tables":{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.rglob("*.csv")}}
    (root/"qa/peer_output_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":main()
