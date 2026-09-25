"""'Sibling group' rule: >=2 records of one S1 cluster that replace the S1's number with the SAME new number."""
import collections


def covered(x, others):
    for o in others:
        if x == o or (len(x) >= 2 and len(o) >= 2 and (x.startswith(o) or o.startswith(x) or x.endswith(o) or o.endswith(x))):
            return True
    return False


def new_numbers(sn, qn):
    """numbers of q that replace S1 numbers (empty if no replacement)."""
    a, b = sn.split(), qn.split()
    if not a or not b:
        return frozenset()
    if not [x for x in a if not covered(x, b)]:
        return frozenset()
    return frozenset(x for x in b if not covered(x, a))


def sibling_flags(s_nums, members):
    """members: list of (qid, q_nums). Returns set of qids flagged as sibling-group members."""
    news = {qid: new_numbers(s_nums, qn) for qid, qn in members}
    cnt = collections.Counter(x for v in news.values() for x in v)
    return {qid for qid, v in news.items() if any(cnt[x] >= 2 for x in v)}
