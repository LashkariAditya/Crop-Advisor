"""Choose the least-cost purchasable pack combination for a required quantity."""
from __future__ import annotations
import math

def cheapest_pack_mix(required_quantity: float, packs: list[dict]) -> dict:
    if required_quantity < 0: raise ValueError("required quantity cannot be negative")
    if required_quantity == 0: return {"quantity":0,"cost":0,"packs":[]}
    if not packs or any(float(p["quantity"])<=0 or float(p["price"])<0 for p in packs): raise ValueError("packs require positive quantities and nonnegative prices")
    # Bounded dynamic programming in small quantity steps. Quantities are converted to thousandths.
    scale=1000; target=math.ceil(required_quantity*scale); sizes=[max(1,round(float(p['quantity'])*scale)) for p in packs]; dp=[(float('inf'),None)]*(target+max(sizes)); dp[0]=(0,None)
    for q in range(target):
        if not math.isfinite(dp[q][0]): continue
        for i,(size,p) in enumerate(zip(sizes,packs)):
            nxt=q+size; cost=dp[q][0]+float(p['price'])
            if cost<dp[nxt][0]: dp[nxt]=(cost,(q,i))
    best=min(range(target,len(dp)),key=lambda q:dp[q][0]); counts=[0]*len(packs); q=best
    while q:
        prev,i=dp[q][1]; counts[i]+=1; q=prev
    return {"quantity":sum(c*float(p['quantity']) for c,p in zip(counts,packs)),"required_quantity":required_quantity,"cost":dp[best][0],"packs":[{"pack":p,"count":c} for c,p in zip(counts,packs) if c]}
