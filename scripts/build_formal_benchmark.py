"""Freeze evaluation data before inspecting model predictions."""
import csv,json,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/formal-benchmark-20260918'
rng=random.Random(20260918);cases=[]
def add(group,idx,context,q,label,source='constructed policy case'):
    if q['type']=='choice':
        items=list(q['criteria'].items());rng.shuffle(items);q={**q,'criteria':dict(items)}
    cases.append({'id':f'{group}_{idx:03d}','group':group,'context':context,'questions':{'decision':q},'expected':{'decision':label},'source':source})
bank={
 'card_arrival':('delivery','Waiting for the original physical card to arrive'),
 'activate_my_card':('activation','How to activate a received card'),
 'lost_or_stolen_card':('lost','A physical payment card is lost or stolen'),
 'change_pin':('pin','Changing the PIN for a payment card'),
 'transaction_charged_twice':('duplicate','The same payment was charged twice'),
 'declined_card_payment':('declined','A card purchase was declined or rejected'),
 'pending_transfer':('pending','A bank transfer remains pending'),
 'terminate_account':('closure','Closing or terminating the bank account'),
 'cash_withdrawal_charge':('withdrawal','Fees charged for withdrawing cash at an ATM'),
 'request_refund':('refund','Requesting a refund or learning how to get a refund')}
rows=list(csv.DictReader((OUT/'banking77-test.csv').open(encoding='utf-8')))
criteria=dict(bank.values())
for category,(label,_) in bank.items():
    selected=rng.sample([(i,r) for i,r in enumerate(rows) if r['category']==category],10)
    for i,r in selected:add('banking',i,r['text'],{'type':'choice','instructions':'Which banking intent best matches the customer message?','criteria':criteria},label,{'dataset':'BANKING77 test subset','row_index':i,'original_label':category})
fast=["Say good morning in French.","What is the capital of Japan?","Convert 3 kilometers to meters.","Give a synonym for happy.","Write a polite one-sentence thank-you.","Name three primary colors.","Translate cat into Spanish.","What day follows Tuesday?","Write a short birthday greeting.","Correct the spelling of recieve.","Summarize this in five words: the meeting moved from Monday to Friday.","Is a whale a mammal?","Give the plural of child.","What is 12 plus 7?","Rewrite I cant attend with correct punctuation.","List two examples of citrus fruit.","Reply yes politely to a dinner invitation."]
coding=["Implement a Python parser for nested arithmetic expressions.","Fix the JavaScript promise chain that swallows rejected errors.","Write SQL to find customers with no orders.","Review this Rust function for ownership mistakes.","Generate unit tests for a CSV import function.","Implement a C++ thread-safe queue.","Debug a CUDA matrix multiplication kernel.","Write a TypeScript REST API with input validation.","Refactor this Python class to remove duplicate code.","Create a Bash script to rotate log files.","Implement Dijkstra's algorithm in Go.","Fix a React component that rerenders infinitely.","Write a regular expression and Python code to validate invoice IDs.","Create a database migration adding an indexed timestamp column.","Implement a Java LRU cache.","Write a Dockerfile for a small Node server.","Patch a race condition in this multithreaded C program."]
reasoning=["Prove by induction that the sum of the first n odd numbers is n squared.","Derive a formula for the expected stopping time of this stochastic process.","Construct a multi-stage research plan with resource dependencies and tradeoffs.","Analyze a scheduling puzzle with four teams and conflicting constraints.","Explain whether a causal claim follows from an observational study, considering confounders.","Develop a rigorous argument for the existence of infinitely many primes.","Solve a probability puzzle involving conditional information and dependent draws.","Compare three infrastructure strategies with competing long-term constraints, without writing code.","Derive the optimal policy for a finite-horizon decision problem.","Find a counterexample to a proposed theorem about all continuous functions.","Work through a multi-step geometry proof involving similar triangles.","Reason about equilibrium in a game where each player can change strategy.","Analyze a hypothetical contract dispute using only the supplied rules, showing the logical dependencies.","Develop a detailed disaster-recovery plan balancing staffing, costs, and dependencies.","Solve a logic puzzle with five houses and multiple relational clues.","Evaluate whether an argument is valid and construct a formal proof or counterexample."]
route={'type':'choice','instructions':'Choose the model according to these capability rules. Requests to write, debug, or review program code go to coding, even when reasoning is also needed. Complex proofs, multi-step reasoning, or planning without code go to reasoning. Straightforward language, factual, or simple arithmetic tasks go to fast.','criteria':{'fast':'Simple language, factual, and straightforward arithmetic tasks','reasoning':'Complex proofs, multi-step reasoning, or planning that does not request program code','coding':'Writing, debugging, testing, or reviewing program code'}}
for label,texts in [('fast',fast),('coding',coding),('reasoning',reasoning)]:
 for text in texts:add('routing',sum(c['group']=='routing' for c in cases),text,route,label)
boolean_groups=[
 ('Does the CURRENT customer explicitly request a refund in their latest statement? Quoted third-party requests, withdrawn requests, and questions about policy alone do not count.',
  ['Please refund the duplicate payment.','I want my money back for this purchase.','Return the amount you charged me in error.','My latest request is a refund, please process it.','Please issue a refund for order 482.'],
  ['I do not want a refund.','What is the general refund policy?','My friend said "please refund me"; I only need a receipt.','Earlier I requested a refund, but now I withdraw that request.','The charge looks correct; thanks for checking.']),
 ('Does the CURRENT customer explicitly ask to speak to a human agent?',
  ['Connect me to a human agent.','Please let me speak with a real person.','I want a human support representative.','Transfer this chat to a staff member, please.','Can a person from your support team talk to me?'],
  ['I do not need a human agent.','Your automated answer resolved it.','My colleague requested a human; I have no such request.','The word human appears in the product description.','Please send the receipt by email.']),
 ('Does the latest statement explicitly confirm the parcel has been delivered to the customer? Predictions and denials do not count.',
  ['I received my parcel today.','The package was delivered to me this morning.','My order has arrived and is in my hands.','I confirm receipt of the delivery.','The courier handed me the parcel.'],
  ['My parcel has not arrived.','It should arrive tomorrow.','The tracking says out for delivery.','My neighbor got their package, but mine is missing.','Has my parcel been delivered yet?']),
 ('Does the latest customer statement explicitly request cancellation of their subscription?',
  ['Cancel my subscription immediately.','Please stop my recurring subscription.','I would like to terminate my subscription.','End my subscription at the next billing date.','Do not renew my subscription; cancel it.'],
  ['Do not cancel my subscription.','What happens if someone cancels?','I cancelled a different service, not this one.','Keep my subscription active.','Please upgrade my subscription.']),
 ('Does the person explicitly grant permission to receive marketing emails in their latest statement?',
  ['I agree to receive marketing emails.','You may email me promotional offers.','Please subscribe me to your marketing mailing list.','I consent to receiving your promotional emails.','Yes, send me product offers by email.'],
  ['Do not send marketing emails.','Send only my order receipt.','I previously consented, but I now revoke that consent.','My coworker agreed to marketing emails; I have not.','What kinds of marketing emails do you send?'])]
for instructions,positive,negative in boolean_groups:
 for label,texts in [('yes',positive),('no',negative)]:
  for text in texts:add('boolean',sum(c['group']=='boolean' for c in cases),text,{'type':'noul','instructions':instructions},label)
levels=['Information only: no fault or loss is reported','Cosmetic issue only: appearance is wrong but functionality works','One customer has a malfunction but can complete the task using a workaround','One customer is completely blocked, with no workaround','Multiple customers are completely blocked']
score={'type':'score','instructions':'Assign impact using the rubric. Use only the stated affected population and ability to complete the task; do not infer unreported outages.','criteria':levels}
products=['calendar','invoice portal','booking page','report viewer','checkout','file upload','dashboard','account editor','search page','message center']
for level in range(5):
 for product in products:
  descriptions=[f'An individual asks how to use the {product}. They report no malfunction or loss.',f'The {product} has a misaligned icon. All functions work normally; this is only cosmetic.',f'For one customer, a button in the {product} fails. They can complete the same task through the menu workaround.',f'One customer cannot use the {product} at all and has no workaround. Everyone else can use it.',f'The {product} is entirely unavailable to multiple customers, and none can complete their tasks.']
  add('scoring',sum(c['group']=='scoring' for c in cases),descriptions[level],score,str(level))
assert len(cases)==250
# Stratified interleaving ensures the first 20 include all four groups.
groups={k:[c for c in cases if c['group']==k] for k in ['banking','routing','boolean','scoring']}
for group in groups.values():rng.shuffle(group)
ordered=[]
while any(groups.values()):
 for g in groups.values():
  if g:ordered.append(g.pop())
robust=[]
for base in ordered[:12]:
 c=json.loads(json.dumps(base));c['id']='reordered_'+base['id'];c['source_case']=base['id'];q=c['questions']['decision']
 if q['type']=='choice':q['criteria']=dict(reversed(list(q['criteria'].items())))
 elif q['type']=='noul':q['instructions']='Evaluate the following yes/no criterion carefully: '+q['instructions']
 else:q['instructions']='Read the impact rubric and select its best-matching level. '+q['instructions']
 robust.append(c)
# Explicit token-collision stress, separated from core quality scores.
for i,kind in enumerate(['billing','shipping','returns']):
 robust.append({'id':f'collision_{kind}','group':'collision_stress','context':{'billing':'My card was charged twice.','shipping':'My package never arrived.','returns':'The shoes do not fit; I want to exchange them.'}[kind], 'questions':{'decision':{'type':'choice','instructions':'Which department handles this issue?','criteria':{'TEAM_BILLING':'Payment errors and duplicate charges','TEAM_SHIPPING':'Missing deliveries','TEAM_RETURNS':'Product returns and exchanges'}}},'expected':{'decision':'TEAM_'+kind.upper()},'source':'constructed first-token collision stress'})
manifest={'seed':20260918,'cases':ordered,'robustness_cases':robust,'source':json.loads((OUT/'dataset-source.json').read_text()),'notes':['BANKING77 is a fixed 10-intent balanced subset; results are not full BANKING77 scores.','150 policy cases are constructed, including templated severity cases; not independent real-world samples.','Pilot is the first 20 frozen cases; no prompt tuning after viewing benchmark outputs.','Earlier debugging examples are excluded; CPU/GPU source and data hashes are recorded.']}
path=OUT/'cases.json'
if path.exists():raise SystemExit('Frozen cases already exist; refusing to overwrite')
path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(OUT/'cases.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n',encoding='utf-8')
print('Frozen',len(ordered),'quality cases and',len(robust),'robustness cases')
