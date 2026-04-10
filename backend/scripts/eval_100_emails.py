#!/usr/bin/env python3
"""CustomerCare eval — compare drafts vs colleague answers from SentItems.
Extracts question from quoted text in sent reply body.
Usage: python3 scripts/eval_100_emails.py [--limit 100] [--days 30]"""
from __future__ import annotations
import asyncio, json, math, re, sys, time
from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
CC_URL = "http://localhost:8000"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
OBSIDIAN_REPORTS = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/PARA/!inbox/!reports"
from app.email.auth import get_auth_headers
from app.config import settings
INTERNAL_DOMAINS = {"neuzrt.hu", "nffku.hu", "nffku.onmicrosoft.com"}

def _html_to_text(html): return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
def _text_sim(a, b): return SequenceMatcher(None, a.lower(), b.lower()).ratio() if a and b else 0.0

async def _sem_sim(a, b):
    if not a or not b: return 0.0
    try:
        from app.rag.embeddings import embed_texts
        embs = embed_texts([a[:1000], b[:1000]])
        if len(embs) != 2: return _text_sim(a, b)
        dot = sum(x*y for x,y in zip(embs[0], embs[1]))
        na, nb = math.sqrt(sum(x*x for x in embs[0])), math.sqrt(sum(x*x for x in embs[1]))
        return dot/(na*nb) if na and nb else 0.0
    except: return _text_sim(a, b)

def _split_reply_quoted(body):
    markers = [r"^From:\s", r"^Sent:\s", r"^Feladó:\s", r"^Küldés ideje:\s", r"^_{10,}", r"^-{5,}$"]
    lines = body.split("\n")
    idx = len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if any(re.match(p, s, re.I) for p in markers): idx = i; break
    reply = "\n".join(lines[:idx]).strip()
    quoted_lines = lines[idx:]
    body_start = 0
    for j, ql in enumerate(quoted_lines):
        qs = ql.strip()
        if qs.startswith(("From:","Sent:","To:","Subject:","Cc:","Feladó:","Küldés ideje:","Címzett:","Tárgy:")): body_start = j+1
        elif body_start > 0 and qs: break
    return reply, "\n".join(quoted_lines[body_start:]).strip()

async def run(limit=100, days=30):
    mailbox = settings.shared_mailboxes.split(",")[0].strip()
    headers = get_auth_headers()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Fetch sent replies
    all_emails, seen_ids = [], set()
    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/SentItems/messages"
    params = {"$filter": f"sentDateTime ge {since}", "$orderby": "sentDateTime desc", "$top": "100", "$select": "id,subject,from,toRecipients,body,sentDateTime"}
    async with httpx.AsyncClient(timeout=60) as client:
        pages = 0
        while pages < 10:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200: break
            data = resp.json()
            for msg in data.get("value", []):
                mid = msg.get("id","")
                if mid and mid not in seen_ids: seen_ids.add(mid); all_emails.append(msg)
            nl = data.get("@odata.nextLink")
            if not nl: break
            url, params, pages = nl, {}, pages+1

    replies = [e for e in all_emails if (e.get("subject") or "").lower().startswith("re:")]
    replies = [e for e in replies if e.get("toRecipients") and e["toRecipients"][0].get("emailAddress",{}).get("address","").split("@")[-1].lower() not in INTERNAL_DOMAINS]
    print(f"Mailbox: {mailbox} | Sent replies: {len(replies)}")

    results, skipped, errors = [], 0, 0
    for sent in replies:
        if len(results) >= limit: break
        subject = sent.get("subject","(none)")
        body = _html_to_text(sent.get("body",{}).get("content",""))
        colleague, question = _split_reply_quoted(body)
        if len(question) < 20 or len(colleague) < 10: skipped += 1; continue
        to = sent.get("toRecipients",[{}])[0].get("emailAddress",{})
        n = len(results)+1
        print(f"[{n}/{limit}] {subject[:55]}...", end=" ", flush=True)
        try:
            t0 = time.time()
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"{CC_URL}/draft/generate", json={"email_text":question[:3000],"email_subject":subject.replace("RE: ",""),"sender_name":to.get("name",""),"sender_email":to.get("address",""),"app_ids":list(set(re.findall(r"OETP-\d{4}-\d{4,8}",question,re.I))),"top_k":5,"max_context_chunks":3})
                hr = r.json() if r.status_code==200 else {}
            dur = time.time()-t0
        except Exception as e: print(f"ERROR ({e})"); errors+=1; continue
        if hr.get("skip"): print(f"SKIP"); skipped+=1; continue
        ht = _html_to_text(hr.get("body_html",""))
        ss = await _sem_sim(ht, colleague); ts = _text_sim(ht, colleague); sim = ss*0.7+ts*0.3
        from app.reasoning.style_score import compute_style_score
        st = compute_style_score(ht, colleague, to.get("name",""))
        status = "MATCH" if sim>=0.5 else "PARTIAL" if sim>=0.25 else "MISMATCH"
        print(f"{status} (sem={ss:.2f} style={st['overall']:.2f} {dur:.1f}s)")
        results.append({"subject":subject[:80],"question":question[:300],"colleague_response":colleague[:500],"cc_response":ht[:500],"similarity":round(sim,3),"semantic_sim":round(ss,3),"text_sim":round(ts,3),"style_score":st["overall"],"style_components":st["components"],"confidence":hr.get("confidence","?"),"status":status,"llm_provider":hr.get("llm_provider","?"),"duration_s":round(dur,1)})
        await asyncio.sleep(0.3)

    if not results: print("No results."); return
    sims = [r["similarity"] for r in results]
    avg_sim, match, partial, mismatch = sum(sims)/len(sims), sum(1 for r in results if r["status"]=="MATCH"), sum(1 for r in results if r["status"]=="PARTIAL"), sum(1 for r in results if r["status"]=="MISMATCH")
    avg_style = sum(r["style_score"] for r in results)/len(results)
    print(f"\n{'='*60}\nRESULTS: {len(results)} tested, {skipped} skipped, {errors} errors\nAverage similarity: {avg_sim:.3f}\nMATCH: {match} ({match/len(results):.0%}) | PARTIAL: {partial} | MISMATCH: {mismatch}\nAverage style: {avg_style:.3f}\nConfidence: {dict(Counter(r['confidence'] for r in results))}")
    # Style components
    for k in ["greeting","length","closing","formality","brevity"]:
        vals = [r.get("style_components",{}).get(k,0) for r in results]
        print(f"  {k}: {sum(vals)/len(vals):.2f}")

    now = datetime.now(timezone.utc)
    report = {"eval_date":now.isoformat(),"total_tested":len(results),"avg_similarity":round(avg_sim,3),"match_rate":round(match/len(results),3),"avg_style":round(avg_style,3),"details":results}
    (Path(__file__).parent.parent/"data"/"eval_100_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"JSON saved.")

if __name__ == "__main__":
    import argparse; p = argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=100); p.add_argument("--days",type=int,default=30); a=p.parse_args()
    asyncio.run(run(limit=a.limit, days=a.days))
