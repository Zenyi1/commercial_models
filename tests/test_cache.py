"""Write-through research cache — pay once per question."""

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.cache import CachingProvider, ResearchCache, cache_key
from valuation_engine.research.provider import ResearchProvider, ResearchQuery


class _Counting(ResearchProvider):
    name = "counting"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def available(self):
        return True

    def get(self, query):
        self.calls += 1
        return self.answer


def _q():
    return ResearchQuery(key="net_price_usd", territory_id="mexico", asset_id="A1")


def test_cache_key_is_stable():
    assert cache_key(_q()) == "A1|mexico|net_price_usd"


def test_caches_success_and_skips_inner_on_hit(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    inner = _Counting(SourcedValue(value=1200.0, kind="point"))
    prov = CachingProvider(inner, cache)
    a = prov.get(_q())
    b = prov.get(_q())
    assert a.value == 1200.0 and b.value == 1200.0
    assert inner.calls == 1  # second call served from cache


def test_caches_none_so_it_is_not_re_paid(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    inner = _Counting(None)
    prov = CachingProvider(inner, cache)
    assert prov.get(_q()) is None
    assert prov.get(_q()) is None
    assert inner.calls == 1  # a definitive None is cached, not re-queried


def test_refresh_bypasses_cache_read(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    inner = _Counting(SourcedValue(value=5.0, kind="point"))
    CachingProvider(inner, cache).get(_q())          # seed
    CachingProvider(inner, cache, refresh=True).get(_q())  # forces fresh call
    assert inner.calls == 2


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.json"
    inner = _Counting(SourcedValue(value=7.0, kind="point"))
    CachingProvider(inner, ResearchCache(path)).get(_q())
    # New cache object reading the same file -> still a hit, inner not called again
    inner2 = _Counting(SourcedValue(value=99.0, kind="point"))
    sv = CachingProvider(inner2, ResearchCache(path)).get(_q())
    assert sv.value == 7.0 and inner2.calls == 0


def test_namespaces_prevent_tier_collision(tmp_path):
    """A fast-tier cached None (search ns) must not occupy the deep tier's key,
    or DeepResearch would be short-circuited and never run on the residual."""
    cache = ResearchCache(tmp_path / "c.json")
    fast = CachingProvider(_Counting(None), cache, namespace="search")
    assert fast.get(_q()) is None
    assert cache.get(cache_key(_q(), "search")) is not None   # fast None cached here
    assert cache.get(cache_key(_q(), "deep")) is None          # deep slot still empty
