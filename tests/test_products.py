import pytest
from backend.adapters.products import mock_products,sort_by_unit_price
from backend.services.pack_math import cheapest_pack_mix

def test_mock_products_have_timestamp_link_and_sort():
    rows=sort_by_unit_price(mock_products('Rice'))
    assert all(x['mock'] and x['last_updated'] and x['url'] for x in rows)
    assert rows[0]['unit_price_inr']<=rows[-1]['unit_price_inr']

def test_cheapest_pack_combination():
    r=cheapest_pack_mix(2.2,[{'quantity':1,'price':10},{'quantity':2.5,'price':20}])
    assert r['quantity']>=2.2
    assert r['cost']==20

def test_invalid_pack_rejected():
    with pytest.raises(ValueError): cheapest_pack_mix(1,[{'quantity':0,'price':1}])

def test_zero_quantity_needs_no_pack():
    assert cheapest_pack_mix(0,[])['cost']==0

def test_exact_pack_quantity_does_not_add_extra_packs():
    r=cheapest_pack_mix(2,[{'quantity':1,'price':5}])
    assert r['quantity']==2
    assert r['cost']==10
