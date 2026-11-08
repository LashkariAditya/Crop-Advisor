import pytest
from backend.services.economics import economics_report

def test_acre_conversion_and_basic_profit():
    r=economics_report('rice',1,'acre',{'costs':{'seed':10000},'yield_kg_per_ha':1000,'price_inr_per_kg':20})
    assert r['area_hectares']==pytest.approx(.40468564224)
    assert r['total_cost_inr']==pytest.approx(4046.8564224)
    assert r['revenue_inr']==pytest.approx(8093.7128448)
    assert r['net_profit_inr']==pytest.approx(r['revenue_inr']-r['total_cost_inr'])

def test_cost_inputs_cannot_be_negative():
    with pytest.raises(ValueError): economics_report('rice',1,'acre',{'costs':{'seed':-1}})

def test_risk_requires_all_inputs():
    r=economics_report('rice',1,'acre',{'risk_inputs':{'price_volatility_0_1':.4}})
    assert r['risk_score'] is None
    r=economics_report('rice',1,'acre',{'risk_inputs':{'price_volatility_0_1':.4,'yield_variance_0_1':.5,'weather_risk_0_1':.2}})
    assert r['risk_score']==pytest.approx(37.5)
