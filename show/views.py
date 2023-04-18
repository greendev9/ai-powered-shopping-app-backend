from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image, ImageDraw
from PIL import ImageCms
from base64 import b64encode
import base64
from io import BytesIO
from torch import autocast
import requests
import PIL
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
import io
from PIL import Image, ImageDraw
from PIL import ImageCms
import json
from django.http import JsonResponse
from serpapi import GoogleSearch
from django.templatetags.static import static
import random
import logging

logger = logging.getLogger('django')

device = "cuda"
# pipe = DiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     # "runwayml/stable-diffusion-v1-5",
#     #"CompVis/stable-diffusion-v1-4",
#     revision="fp16", 
#     torch_dtype=torch.float16,
#     use_auth_token=True
# ).to(device)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(device)

# pipe = StableDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2",
#     torch_dtype=torch.float16,
# ).to(device)

def image_padding(image):
    # image paading
    new_size = max(image.width, image.height)
    image_padded = Image.new(image.mode, (new_size, new_size), (0, 0, 0))
    left_pad = (int)((new_size-image.width)/2)
    top_pad = (int)((new_size-image.height)/2)
    image_padded.paste(image, (left_pad, top_pad))
    return (image_padded, left_pad, top_pad)

@csrf_exempt
def index(request):
    if request.method == "POST":
        logger.debug('---showonme-image processing started')

        data = json.loads(request.body.decode())
        image_base64 = data['originImage'].replace("data:image/png;base64,", "")
        image_base64 = image_base64.replace("data:image/jpeg;base64,", "")
        bytes_decoded = base64.b64decode(image_base64)
        try:
            init_img = Image.open(BytesIO(bytes_decoded))
        except PIL.UnidentifiedImageError:
            print("PIL.UnidentifiedImageError")
            return JsonResponse({'response_code':False})
            # return HttpResponse("PIL.UnidentifiedImageError")
        
        init_img.save('init.png')

        # change percent to pixel
        data['cropRect']['x'] = int(init_img.width * data['cropRect']['x'] / 100)
        data['cropRect']['y'] = int(init_img.height * data['cropRect']['y'] / 100)
        data['cropRect']['width'] = int(init_img.width * data['cropRect']['width'] / 100)
        data['cropRect']['height'] = int(init_img.height * data['cropRect']['height'] / 100)

        crop_padding = 32
        crop_region = {}
        crop_region['x1'] = (int)(max(data['cropRect']['x'] - crop_padding, 0))
        crop_region['y1'] = (int)(max(data['cropRect']['y'] - crop_padding, 0))
        crop_region['x2'] = (int)(min(data['cropRect']['x'] + data['cropRect']['width'] + crop_padding, init_img.width))
        crop_region['y2'] = (int)(min(data['cropRect']['y'] + data['cropRect']['height'] + crop_padding, init_img.height))
        print('crop_region')
        print(crop_region)

        crop_image = init_img.crop(tuple(crop_region.values()))
        crop_image.save('crop_image.png')
        emebed_size = max(crop_image.width, crop_image.height)

        # mask_image = Image.new("RGB", (crop_image.width, crop_image.height), (255, 255, 255))
        mask_image = Image.new("RGB", (init_img.width, init_img.height), (0, 0, 0))
        img_draw = ImageDraw.Draw(mask_image)
        w, h = int(data['cropRect']['width']), int(data['cropRect']['height'])
        x = data['cropRect']['x']
        y = data['cropRect']['y']
        shape = [(x, y), (x+w, y+h)]
        img_draw.rectangle(shape, fill ="#ffffff")
        mask_image = mask_image.crop(tuple(crop_region.values()))
        mask_image.save('mask.png')

        crop_height=crop_image.height
        crop_width=crop_image.width
        crop_image, left_pad, top_pad = image_padding(crop_image)
        mask_image, left_pad, top_pad = image_padding(mask_image)
        crop_image.save('crop_padding.png')
        mask_image.save('mask_padding.png')
        padded_size = crop_image.width
        crop_image = crop_image.convert("RGB").resize((512, 512))
        mask_image = mask_image.convert("RGB").resize((512, 512))

        output = pipe(prompt=data['prompt'], image=crop_image, mask_image=mask_image, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1, height=512, width=512)
        result_image = output.images[0]
        result_image.save('result.png')

        result_image = result_image.convert("RGB").resize((padded_size, padded_size))
        result_image = result_image.crop((left_pad, top_pad, left_pad+crop_width, top_pad+crop_height))
        edge_padding = 1
        result_image = result_image.crop((edge_padding, edge_padding, result_image.width-edge_padding, result_image.height-edge_padding))
        result_filename = str(random.randint(0,10000))+'.png'
        result_image.save('build/image/'+result_filename)
        result_image.save('result_crop.png')

        init_img.paste(result_image, (crop_region['x1']+edge_padding, crop_region['y1']+edge_padding))

        image_io = BytesIO()
        init_img.save(image_io, 'PNG')
        dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
        response = {"data_url":dataurl, "result_filename":result_filename}
        
        logger.debug('---showonme-image processing finished')
        return HttpResponse(json.dumps(response), content_type="application/json")

@csrf_exempt
def get_images(request):
    data = json.loads(request.body.decode())
    # google image search
    result_filename = data['result_filename']
    
    params = {
        # "engine": "google_reverse_image",
        # "image_url": "http://207.246.90.80/image/"+result_filename,
        "engine": "google_lens",
        "url": "http://207.246.90.80/image/"+result_filename,
        "api_key": "565119a35665cffb3eda2fefe7de05c6d95cd3ac8af3ebb8b4eae7a0994e43ce",
        "num": 12,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    # shopping_results = results["visual_matches"]
    # if (results["visual_matches"]):
    #     shopping_results = results["visual_matches"]
    # else:
    #     shopping_results = results
    shopping_results = results

    # params = {
    #     "engine": "google_shopping",
    #     "q": results["search_information"]["query_displayed"],
    #     "api_key": "565119a35665cffb3eda2fefe7de05c6d95cd3ac8af3ebb8b4eae7a0994e43ce",
    #     "num": 12,
    # }
    # search = GoogleSearch(params)
    # results = search.get_dict()
    # shopping_results = results["shopping_results"]
    # return JsonResponse(shopping_results, safe=False)

    response = {"shopping_results":shopping_results}
    return HttpResponse(json.dumps(response), content_type="application/json")