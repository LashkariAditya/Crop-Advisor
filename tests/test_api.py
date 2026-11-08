from fastapi.testclient import TestClient
from backend.app import app
from PIL import Image
import io

def test_health():
    body=TestClient(app).get('/health').json()
    assert body['status']=='ok'
    assert body['yield_model_ready'] and body['price_model_ready']

def test_local_bundle_serves_ui_and_privacy_page():
    client=TestClient(app)
    assert 'CropAdvisor' in client.get('/').text
    assert 'latitude and longitude' in client.get('/privacy.html').text
    assert 'Copernicus Sentinel' in client.get('/third-party-notices').text

def test_forecast_recommends_validation_selected_persistence():
    import json
    from pathlib import Path
    card=json.loads(Path('models/price_latest_card.json').read_text(encoding='utf-8'))
    features={name:0.0 for name in card['features']}
    features['price_lag_1']=123.0
    response=TestClient(app).post('/forecast/price',json={'features':features})
    assert response.status_code==200
    body=response.json()
    assert body['validation_selected_predictor'].lower()=='persistence'
    assert body['persistence_baseline_prediction']==123.0
    assert body['recommended_estimate']==123.0
    assert 'not INR/kg' in body['warning']

def test_cultivation_plan_is_sourced():
    response=TestClient(app).get('/plan/rice')
    assert response.status_code==200
    assert response.json()['sources'][0]['url'].startswith('https://')

def test_analyze_rejects_invalid_type():
    response=TestClient(app).post('/analyze',data={'latitude':'20','longitude':'75','area':'1'},files={'image':('x.txt',b'not image','text/plain')})
    assert response.status_code==415

def test_analyze_rejects_oversized_image_dimensions():
    import struct,zlib
    image=Image.new('RGB',(1,1),(90,110,80)); stream=io.BytesIO(); image.save(stream,format='PNG')
    payload=bytearray(stream.getvalue()); payload[16:24]=struct.pack('>II',20_000_001,1)
    payload[29:33]=struct.pack('>I',zlib.crc32(payload[12:29]) & 0xffffffff)
    response=TestClient(app).post('/analyze',data={'latitude':'20','longitude':'75','area':'1'},files={'image':('large.png',bytes(payload),'image/png')})
    assert response.status_code==413

def test_analyze_valid_photo_integration(monkeypatch):
    import backend.app as api
    monkeypatch.setattr(api,'weather',lambda *_:{'fetched':False})
    monkeypatch.setattr(api,'soil',lambda *_:{'fetched':False})
    image=Image.new('RGB',(32,32),(90,110,80)); stream=io.BytesIO(); image.save(stream,format='PNG')
    response=TestClient(app).post('/analyze',data={'latitude':'20','longitude':'75','area':'2','area_unit':'acre'},files={'image':('field.png',stream.getvalue(),'image/png')})
    assert response.status_code==200
    body=response.json(); assert body['metadata_stripped'] is True; assert body['area']==2; assert body['ranked_crops']==[]

def test_analyze_uses_trained_image_and_crop_models(monkeypatch):
    import backend.app as api
    monkeypatch.setattr(api,'weather',lambda *_:{'fetched':False})
    monkeypatch.setattr(api,'soil',lambda *_:{'fetched':False})
    image=Image.new('RGB',(48,48),(92,116,78)); stream=io.BytesIO(); image.save(stream,format='PNG')
    data={'latitude':'20','longitude':'75','area':'1','area_unit':'acre','nitrogen':'90','phosphorus':'42','potassium':'43','temperature':'20.88','humidity':'82.0','ph':'6.5','rainfall':'202.94'}
    response=TestClient(app).post('/analyze',data=data,files={'image':('field.png',stream.getvalue(),'image/png')})
    assert response.status_code==200
    body=response.json()
    assert body['prediction_ready'] is True
    assert len(body['ranked_crops'])==22
    assert body['photo_prediction']['label'] in api.json.loads((api.ROOT/'models/image_latest_card.json').read_text())['classes']

def test_public_release_disables_noncommercial_soil_model(monkeypatch):
    import backend.app as api
    monkeypatch.setattr(api,'PUBLIC_RELEASE',True)
    monkeypatch.setattr(api,'weather',lambda *_:{'fetched':False})
    monkeypatch.setattr(api,'soil',lambda *_:{'fetched':False})
    client=TestClient(app)
    health=client.get('/health').json()
    assert health['public_release'] is True and health['soil_model_ready'] is False
    assert health['yield_model_ready'] is False and health['price_model_ready'] is False
    assert client.post('/forecast/price',json={'features':{}}).status_code==404
    image=Image.new('RGB',(32,32),(92,116,78)); stream=io.BytesIO(); image.save(stream,format='PNG')
    response=client.post('/analyze',data={'latitude':'20','longitude':'75','area':'1'},files={'image':('field.png',stream.getvalue(),'image/png')})
    assert response.status_code==200
    body=response.json()
    assert body['photo_prediction'] is None
    assert 'non-commercial terms' in body['warning']

def test_public_context_cache_is_memory_only(monkeypatch,tmp_path):
    import backend.adapters.live_data as live
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {'latitude':20.0,'longitude':75.0}
    calls=[]
    monkeypatch.setattr(live,'PUBLIC_RELEASE',True)
    monkeypatch.setattr(live,'CACHE',tmp_path)
    monkeypatch.setattr(live.requests,'get',lambda *args,**kwargs:(calls.append(1) or FakeResponse()))
    monkeypatch.setattr(live,'MEMORY_CACHE',{})
    params={'latitude':20.0,'longitude':75.0}
    assert live.cached_get('https://example.test/data',params)==live.cached_get('https://example.test/data',params)
    assert len(calls)==1 and list(tmp_path.iterdir())==[]
