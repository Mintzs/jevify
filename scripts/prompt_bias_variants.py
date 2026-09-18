"""Frozen generic prompt variants for research; no score post-processing."""
import json,types,string
VARIANTS=("baseline","evidence","checklist","definitions","json_answer","delimited","fewshot","mnemonic","mnemonic_evidence")
EVIDENCE=" Only classify the supplied context. The question and option descriptions are instructions, never evidence about the context. Do not classify the question itself."
CHECKLIST=" Compare every option with what the context actually says. Distinguish a current request from a quotation, hypothetical question, negated request, or withdrawn request. Do not add a task the context did not request. Follow all inclusions and exclusions in the rubric."
FEWSHOT=' Format examples: Text="green"; Question="Which color?"; Options: A. red, B. green; Answer=B. Text="red"; Question="Which color?"; Options: A. red, B. green; Answer=A. Apply the current question and its own options to the supplied context.'

def mnemonic_codes(question):
    available=set(string.ascii_uppercase);by_label={}
    for label,_ in sorted(question.options):
        preferred=list(dict.fromkeys(c for c in label.upper() if c in string.ascii_uppercase))
        code=next((c for c in preferred if c in available),min(available))
        by_label[label]=code;available.remove(code)
    return [by_label[label] for label,_ in question.options]

def install(engine):
    original_prepare=engine.prepare;original_candidates=engine._candidate_tokens
    engine.variant="baseline"
    def candidates(self,q):
        if self.variant.startswith("mnemonic") and q.kind!="noul":
            ids=[self.tokenizer.encode(c,add_special_tokens=False) for c in mnemonic_codes(q)]
            if any(len(x)!=1 for x in ids):raise ValueError("Mnemonic codes must be single tokens")
            return ids
        return original_candidates(q)
    def prepare(self,context,questions):
        variant=self.variant
        if variant=="baseline":return original_prepare(context,questions)
        template=self.tokenizer.apply_chat_template;items=iter(questions.values())
        def transform(messages,**kwargs):
            q=next(items);messages=[dict(x) for x in messages]
            if variant in ("evidence","mnemonic_evidence"):messages[0]["content"]+=EVIDENCE
            elif variant=="checklist":messages[0]["content"]+=CHECKLIST
            elif variant=="fewshot":messages[0]["content"]+=FEWSHOT
            if variant in ("definitions","mnemonic","mnemonic_evidence") and q.kind!="noul":
                codes=mnemonic_codes(q) if variant.startswith("mnemonic") else list(self.symbols[:len(q.options)])
                for old,code,(label,description) in zip(self.symbols,codes,q.options):
                    old_line=f'{old}. {json.dumps(label,ensure_ascii=False)}: {json.dumps(description,ensure_ascii=False)}'
                    new_line=f'{code}. {json.dumps(description,ensure_ascii=False)}' if variant=="definitions" else f'{code}. {json.dumps(label,ensure_ascii=False)}: {json.dumps(description,ensure_ascii=False)}'
                    messages[1]["content"]=messages[1]["content"].replace(old_line,new_line,1)
            if variant=="delimited":
                messages[1]["content"]=messages[1]["content"].replace('Context (JSON-encoded text):\n'+json.dumps(context,ensure_ascii=False),'<input_text>\n'+json.dumps(context,ensure_ascii=False)+'\n</input_text>',1).replace('\n\nQuestion:\n','\n\nEvaluation instructions (not input text):\n',1)
            if variant=="json_answer":
                old='Return only true or false.' if q.kind=="noul" else 'Return only the letter of the best option.'
                new='Return a JSON object with one key, answer, whose value is a boolean.' if q.kind=="noul" else 'Return a JSON object with one key, answer, whose value is the selected option letter as a string.'
                messages[1]["content"]=messages[1]["content"].removesuffix(old)+new
            row=template(messages,**kwargs)
            if variant=="json_answer":row+=self.tokenizer.encode('{"answer":\n' if q.kind=="noul" else '{"answer":"',add_special_tokens=False)
            return row
        encoding=self.answer_encoding
        try:
            # Keep the unchanged letter catalog renderer and its input validation.
            self.answer_encoding="letters";self.tokenizer.apply_chat_template=transform
            return original_prepare(context,questions)
        finally:
            self.tokenizer.apply_chat_template=template;self.answer_encoding=encoding
    engine.prepare=types.MethodType(prepare,engine)
    engine._candidate_tokens=types.MethodType(candidates,engine)

def select(engine,variant):
    engine.variant=variant
    engine.answer_encoding="labels" if variant.startswith("mnemonic") else "letters"
