import asyncio
from app.email.feedback import check_feedback

async def run():
    res = await check_feedback('lakossagitarolo@neuzrt.hu', 24)
    modified = [d for d in res['details'] if d['status'] in ('modified', 'heavily_modified')]
    for m in modified:
        print(f"Subject: {m['subject']}")
        print(f"Similarity: {m['similarity']}")
        print(f"Status: {m['status']}")
        print(f"Match Method: {m.get('match_method')}")
        # Need to see how check_feedback gets the texts
        
if __name__ == '__main__':
    asyncio.run(run())
