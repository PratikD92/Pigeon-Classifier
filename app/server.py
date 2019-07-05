import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.googleapis.com/drive/v3/files/1SXtxmr1_vOeGRuRFIdWAVQ-iE2-ET8ts?alt=media&key=AIzaSyAo3sdBL8E-bm8zpgmTg5j1KrZnZZoyRG4'
export_file_name = 'export.pkl'

classes = ['aachen_lacquer_shield_owl', 'african_owl', 'altenburg_trumpeter', 'american_giant_runt', 'american_show_racer', 'anatolian_ringbeater', 'ancient_tumbler', 'antwerp_smerle', 'arabian_trumpeter', 'archangel_pigeon', 'armenian_tumbler', 'australian_performing_tumbler', 'australian_saddleback_tumbler', 'barb_pigeon', 'belgian_ringbeater', 'berlin_long_faced_tumbler', 'berlin_short_faced_tumbler', 'bijeljina_roller', 'birmingham_roller', 'bohemian_fairy_swallow_pigeon', 'bokhara_trumpeter', 'breslau_tumbler', 'british_show_racer', 'brunner_pouter', 'budapest_highflyer', 'budapest_short_faced_tumbler', 'bursa_eumbler', 'english_magpie_pigeon', 'gaditano_pouter', 'galatz_roller', 'german_beauty_homer', 'german_modena', 'german_nun_pigeon', 'ghent_cropper', 'granadino_pouter', 'helmet', 'holle_cropper', 'homing_pigeon', 'ice_pigeon', 'indian_fantail', 'indian_gola', 'italian_owl', 'jacobin', 'kiev_tumbler', 'king_pigeon', 'komorn tumbler', 'k�nigsberg_colour_head_tumbler', 'lahore_pigeon', 'lucerne_gold_collar', 'modena_pigeon', 'norwich_cropper', 'nun pigeon', 'polish_helmet_pigeon', 'rock_dove']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
