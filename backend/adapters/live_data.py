"""Cached public climate and SoilGrids lookups; failures return unavailable data."""
from __future__ import annotations
import hashlib,json,time,os
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'data'/'processed'/'api_cache'; CACHE.mkdir(parents=True,exist_ok=True)
PUBLIC_RELEASE=os.getenv('CROPADVISOR_PUBLIC_RELEASE','').lower() in {'1','true','yes'}
MEMORY_CACHE:dict[str,tuple[float,dict]]={}

def cached_get(url,params,ttl=1800):
    key=hashlib.sha256((url+json.dumps(params,sort_keys=True)).encode()).hexdigest()
    if PUBLIC_RELEASE:
        cached=MEMORY_CACHE.get(key)
        if cached and time.time()-cached[0]<ttl:return cached[1]
        response=requests.get(url,params=params,timeout=20); response.raise_for_status(); data=response.json()
        MEMORY_CACHE[key]=(time.time(),data)
        if len(MEMORY_CACHE)>512:MEMORY_CACHE.pop(next(iter(MEMORY_CACHE)))
        return data
    path=CACHE/(key+'.json')
    if path.exists() and time.time()-path.stat().st_mtime<ttl:
        try:return json.loads(path.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):pass
    response=requests.get(url,params=params,timeout=20); response.raise_for_status(); data=response.json(); path.write_text(json.dumps(data),encoding='utf-8'); return data

def weather(latitude:float,longitude:float):
    """Return current and 7-day daily forecast data from Open-Meteo."""
    try:
        return {"source":"Open-Meteo forecast API","url":"https://open-meteo.com/en/docs","fetched":True,"data":cached_get('https://api.open-meteo.com/v1/forecast',{'latitude':latitude,'longitude':longitude,'current':'temperature_2m,relative_humidity_2m,precipitation','daily':'temperature_2m_mean,precipitation_sum','forecast_days':7,'timezone':'auto'})}
    except (requests.RequestException,ValueError) as exc:return {"source":"Open-Meteo forecast API","fetched":False,"error":str(exc)}

def weather_archive(latitude:float,longitude:float,start_date:str,end_date:str):
    """Fetch historical daily weather from Open-Meteo's archive endpoint."""
    try:return {"source":"Open-Meteo Historical Weather API","url":"https://archive-api.open-meteo.com/v1/archive","fetched":True,"data":cached_get('https://archive-api.open-meteo.com/v1/archive',{'latitude':latitude,'longitude':longitude,'start_date':start_date,'end_date':end_date,'daily':'temperature_2m_mean,precipitation_sum','timezone':'auto'},ttl=86400)}
    except (requests.RequestException,ValueError) as exc:return {"source":"Open-Meteo Historical Weather API","fetched":False,"error":str(exc)}

def soil(latitude:float,longitude:float):
    """Return SoilGrids surface properties when its beta REST service responds."""
    try:
        params={'lon':longitude,'lat':latitude,'property':['phh2o','nitrogen','soc','clay','sand','silt'],'depth':'0-5cm','value':'mean'}
        data=cached_get('https://rest.isric.org/soilgrids/v2.0/properties/query',params,ttl=86400)
        return {"source":"ISRIC SoilGrids v2.0","url":"https://rest.isric.org/soilgrids/v2.0/docs","fetched":True,"data":data}
    except (requests.RequestException,ValueError) as exc:return {"source":"ISRIC SoilGrids v2.0","fetched":False,"error":str(exc)}

def nasa_power(latitude:float,longitude:float,start_date:str,end_date:str):
    """Return daily agroclimate and solar variables from NASA POWER."""
    params={'parameters':'T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN','community':'AG','longitude':longitude,'latitude':latitude,'start':start_date.replace('-',''),'end':end_date.replace('-',''),'format':'JSON'}
    try:return {"source":"NASA POWER","url":"https://power.larc.nasa.gov/docs/services/api/","fetched":True,"data":cached_get('https://power.larc.nasa.gov/api/temporal/daily/point',params,ttl=86400)}
    except (requests.RequestException,ValueError) as exc:return {"source":"NASA POWER","fetched":False,"error":str(exc)}

def _env_value(name:str):
    if name in os.environ:return os.environ[name]
    p=ROOT/'.env'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip().startswith(name+'='):return line.split('=',1)[1].strip().strip('"\'')
    return ''

def mandi_prices(resource_id:str|None=None,filters:dict|None=None):
    """Query a data.gov.in mandi resource when configured, otherwise fail closed."""
    api_key=_env_value('DATA_GOV_IN_API_KEY'); rid=resource_id or _env_value('DATA_GOV_IN_MANDI_RESOURCE_ID')
    if not api_key or not rid:return {"source":"data.gov.in mandi API","fetched":False,"error":"Set DATA_GOV_IN_API_KEY and DATA_GOV_IN_MANDI_RESOURCE_ID in .env after selecting a current official resource."}
    params={'api-key':api_key,'format':'json','limit':100}
    for key,value in (filters or {}).items():params[f'filters[{key}]']=value
    try:return {"source":"data.gov.in mandi API","url":"https://www.data.gov.in/","fetched":True,"data":cached_get(f'https://api.data.gov.in/resource/{rid}',params,ttl=1800)}
    except (requests.RequestException,ValueError) as exc:return {"source":"data.gov.in mandi API","fetched":False,"error":str(exc)}

