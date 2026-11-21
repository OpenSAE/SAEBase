"""Explain candidate relaxation and token-position observations without changing B's RSS result."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from omegaconf import OmegaConf
from saebase.api import SAEIdentity
from saebase.api.neuronpedia import annotate_features
from tasks.capture.operation import neuronpedia_settings

FEATURES=[17410,5436,32506,3044,30221]
TOPICS=['Basketball','Space','Music','Chess','Animals']


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory.resolve()
    audit=json.loads((root/'final_token_audit.json').read_text())
    profiles=json.loads((root/'profiles.json').read_text())
    zs=[np.load(root/f'B_V{i+1}_token_codes.npz')['codes'] for i in range(5)]
    pair=SimpleNamespace(identity=SAEIdentity(**profiles['identity']),sae=SimpleNamespace(cfg=None))
    info,labels=annotate_features(pair,FEATURES,neuronpedia_settings(OmegaConf.create({'task':{}})))
    ends=[r['prefix_last_token_index'] for r in profiles['profiles']]
    observed=[]
    for v,f in enumerate(FEATURES):
        prefix=[float(z[i,f]) for z,i in zip(zs,ends)]
        target=[float(z[-1,f]) for z in zs]
        row={'topic':TOPICS[v],'feature_id':f,'prefix_end_activations':prefix,'target_end_activations':target,'prefix_token_index':ends[v],'target_token_index':profiles['profiles'][v]['target_last_token_index'],'annotation':labels[f]}
        observed.append(row)
    result={'purpose':'Token-position diagnosis only; no replacement of the original final-token RSS table. These are activations, not variant-minus-clean deltas.','device':profiles['device'],'identity':profiles['identity'],'annotation_source':info,'observations':observed,'original_B_final_token_audit':'final_token_audit.json'}
    (root/'position_observations.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# B Grouping: whether the five types of background characteristics can be added to the relaxing rules?','',
           '**Key differences have been identified: five sets of analogues exist at the end of the background sentence and are unique to each other, but four of them end with zero on cooking. The original end of the data is not missing from the candidate threshold; these zero responses cannot be restored only by relaxing the screening rules..**','',
           '## Let’s check the original. B End result','',
           'Original B It’s... 32,768 In a fully-fledged vector, 89 The article feature is at least one. variant Non-zero changes; this 89 It’s all in the original. RSS Front 400 Inner pool. Check the public interpretation of all change features and do not find any other clear themes that are missing. Previous success stories match the current model/SAE Registration status, layer,hook,Resource binding and Neuronpedia The source is the same. No evidence of faulty or faulty storage..','',
           'Here’s the first five known backgrounds. feature Yes.**This time. B Original GPU End of objective**Actual delta,Not to repeat old experiment values.','',
           '| Background | Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Recurrence |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in audit['previous_success_feature_observations_in_B']:
        v=FEATURES.index(row['feature_id']);f=row['feature_id'];d=row['delta'];rec=sum(abs(x)>1e-5 for x in d)/5;url=labels[f]['url']
        lines.append(f"| {TOPICS[v]} | [F{f}]({url}) | "+' | '.join(f'{x:+.5f}' if x else '0' for x in d)+f' | {rec:.1f} |')
    lines += ['','Original B Three cooks. RSS Features(F20700 Increase, F12293/F9190(Reductions)RSS Select the set as it is, not reset rho/kappa Or change. M/K Follow the results..','',
              '## Actual results of the easing of numerical rules','',
              'Relaxing only works on**It’s already been done. RSS Excluded Show Candidates**,No change. RSS In itself. Every one of them is complete. 32,768 We’ll have to check the results before extra cancel. 400 The limit doesn’t miss the non-zero change..','',
              '| Display Conditions | V1 Candidates | V2 | V3 | V4 | V5 |','| --- | --- | --- | --- | --- | --- |']
    names={'strict_single':'Only Ben. variant More than tau','dominance_5':'Ben. variant > Other biggest absolute changes 5 Double','dominance_2':'Ben. variant > Other biggest absolute changes 2 Double','largest_response':'Ben. variant The positive changes are the largest of the five.'}
    for key,groups in audit['rules'].items():lines.append(f"| {names[key]} | "+' | '.join(str(len(g['feature_ids'])) for g in groups)+' |')
    lines += ['','From strict single to five-fold rules, there’s no more candidate. Double the rule brings in extra. V3/F5618(URL/programming questions and answers) and V5/F21846(They can’t be divided into music and animals. They can be further relaxed to ”just ask Ben.”variant Max increases the number of candidates for generic, cross-variant responses, but there is still no clear correspondence between the four missing categories in the original full explanation. Values are less than semantics..','',
              '### It can be used as an item for weaker next-door leads.','',
              'F29390 Yes. V2 It’s a unique increase.+4.74274,Public interpretation“concepts related to physics and energy dynamics”.It is adjacent to a broad physical theme with zero gravity, but it does not clearly identify astronauts or space travel. It can be clearly marked as a physical proximity, not as a link to the original space. feature Matching of equal strength; even if the article is accepted, the rest of the background cannot be filled. It was originally obtained through strict monovariate rules, not by easing the value threshold..','',
              '## Real difference: where the feature is entered','',
              f"Use exactly the same five. B prompts,Same model with L21 SAE This is a metametric diagnostic.{profiles['device']},The model remains as BF16;As a result GPU Intended by other operations, location diagnosis and original GPU The main table is kept separately. The following two locations are from the same diagnosis. forward It’s... native codes,It’s an active value, no. delta,And it’s not running on that table either. RSS.",'',
              '| Background | Feature | Activate against the end of the background sentence | Activate the end of a cooking problem with the same input | Original Neuronpedia explanation |',
              '| --- | --- | --- | --- | --- |']
    for v,row in enumerate(observed):
        ann=row['annotation'];text='; '.join(e['description'] for e in ann['explanations']).replace('|','\\|')
        lines.append(f"| {row['topic']} | [F{row['feature_id']}]({ann['url']}) | {row['prefix_end_activations'][v]:.5f} | {row['target_end_activations'][v]:.5f} | {text} |")
    lines += ['','In this diagnosis, each of the above-mentioned characteristics is**At the end of the other four background sentences**And for all of them. 0:Five categories do form a one-to-one response. At the end of the fixed cooking problem, sports, space, musical instruments, and wildlife are zero, and the chess response is zero..','',
              'CPU Diagnosing the end of the chess set to 0.96941,Original GPU The main table is 0.86217;The specific range of the different back end is different, so this report compares only two positions within the same diagnosis and does not include CPU Replace Value with Original GPU It’s... RSS Table. At the end of the four articles, 0 The facts are consistent in both sets of records..','',
              '## How can we adjust on that?','',
              '1. **Keep original B End of goal RSS caliber**:Three two-way changes in cooking-related characteristics and evidence of exclusion of chess; the clear match remains 1/5.Relaxing the threshold doesn’t translate four zeros into evidence..',
              '2. **Allows an additional independent background location diagnosis**:There are already five clear sets of matches, but they must be marked at the end of the background sentence; these values cannot be added to the original end delta The matrix. We can’t talk about it. RSS Here. recurrence=0.2 We’ve ruled out the five..',
              '3. **If you must have the integrity of the same target position 3+5 Contrast**:Need to re-select an experimental design that can preserve a background response at that location, for example, another SAE Layers;This is a change of research settings that cannot be achieved by renaming candidates or mixing positions.','',
              'No change this time. B Enter, Original RSS Select the set or original master table. The end-to-zero feature is not explained as the model has forgotten the background; only these explanations are given for this layer, this location SAE Feature no longer responds. Not found RSS The filter missing evidence of four non-zero response categories..','',
              '## Full record','',
              '- [Full end vector and relaxation of screening audit](final_token_audit.json)',
              '- [Enter, Index and token Text](profiles.json)',
              '- [Full-activated matrix of five features at the end of the five background and end of the goal](position_observations.json)',
              '- B_V1_token_codes.npz to B_V5_token_codes.npz Keeps all words entered by each word 32768 V codes.',
              '- [Original B Main Table](../rss_banana_ab/B/rss_case.md) · [Original A/B Comparison](../rss_banana_ab/comparison.md)']
    (root/'diagnosis.md').write_text('\n'.join(lines)+'\n')
    with ZipFile(root.parent/'rss_b_diagnosis_report.zip','w',ZIP_DEFLATED) as z:z.write(root/'diagnosis.md','diagnosis.md')
    print(root/'diagnosis.md');print(root.parent/'rss_b_diagnosis_report.zip')


if __name__=='__main__':main()
