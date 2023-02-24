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

@csrf_exempt
def index(request):
    if request.method == "POST":
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
        
        logger.debug('---showonme-step2')
        init_img.save('init.png')

        # change percent to pixel
        data['cropRect']['x'] = init_img.width * data['cropRect']['x'] / 100
        data['cropRect']['y'] = init_img.height * data['cropRect']['y'] / 100
        data['cropRect']['width'] = init_img.width * data['cropRect']['width'] / 100
        data['cropRect']['height'] = init_img.height * data['cropRect']['height'] / 100

        # image paading
        new_size = max(init_img.width, init_img.height)
        init_img_padded = Image.new(init_img.mode, (new_size, new_size), (255, 255, 255))
        left_pad = (int)((new_size-init_img.width)/2)
        top_pad = (int)((new_size-init_img.height)/2)
        right_pad = left_pad + init_img.width
        bottom_pad = top_pad + init_img.height
        init_img_padded.paste(init_img, (left_pad, top_pad))
        init_img_padded.save('init_img_padded.png')
        init_img = init_img_padded

        # create mask image
        img_mask = Image.new("RGB", (init_img.width, init_img.height), (0, 0, 0))
        img_draw = ImageDraw.Draw(img_mask)
        w, h = data['cropRect']['width'], data['cropRect']['height']
        w = int(w)
        h = int(h)
        x = data['cropRect']['x'] + left_pad
        y = data['cropRect']['y'] + top_pad
        shape = [(x, y), (x+w, y+h)]
        print(data['cropRect'])
        print(shape)
        img_draw.rectangle(shape, fill ="#ffffff")
        img_mask.save('mask.png')

        emebed_size = 512

        x_ratio = emebed_size/init_img.width
        y_ratio = emebed_size/init_img.height
        margin = 0
        left = abs(x*x_ratio - margin)
        top = abs(y*y_ratio - margin)
        right = min(left+w*x_ratio + margin, emebed_size)
        bottom = min(top+h*y_ratio + margin, emebed_size)

        init_img = init_img.convert("RGB").resize((emebed_size, emebed_size))
        img_mask = img_mask.convert("RGB").resize((emebed_size, emebed_size))
        init_img.save('init_resize.png')
        img_mask.save('mask_resize.png')

        pipe.safety_checker = lambda images, clip_input: (images, False)
        # output = pipe(prompt=data['prompt'], image=init_img, mask_image=img_mask, num_inference_steps=15, guidance_scale=7.5, num_images_per_prompt=1)
        # output = pipe(prompt=data['prompt'], image=init_img, mask_image=img_mask, num_inference_steps=100, guidance_scale=11, num_images_per_prompt=1)
        output = pipe(prompt=data['prompt'], image=init_img, mask_image=img_mask, num_inference_steps=15, guidance_scale=7.5, num_images_per_prompt=1, height=emebed_size, width=emebed_size)
        output.images[0].save('result.png')

        logger.debug('---showonme-step5')
        result_filename = str(random.randint(0,10000))+'.png'
        output.images[0].crop((left, top, right, bottom)).save('build/image/'+result_filename)

        result_image_total = output.images[0].crop((left_pad*x_ratio, top_pad*y_ratio, right_pad*x_ratio, bottom_pad*y_ratio))

        # image_io_0 = BytesIO()
        # output.images[0].save(image_io_0, 'PNG')
        # dataurl_0 = 'data:image/png;base64,' + b64encode(image_io_0.getvalue()).decode('ascii')
        # image_io_1 = BytesIO()
        # output.images[1].save(image_io_1, 'PNG')
        # dataurl_1 = 'data:image/png;base64,' + b64encode(image_io_1.getvalue()).decode('ascii')
        # image_io_2 = BytesIO()
        # output.images[2].save(image_io_2, 'PNG')
        # dataurl_2 = 'data:image/png;base64,' + b64encode(image_io_2.getvalue()).decode('ascii')
        # response = {"data_url":[dataurl_0, dataurl_1, dataurl_2], "result_filename":result_filename}

        # image_io = BytesIO()
        # output.images[0].save(image_io, 'PNG')
        # dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
        # response = {"data_url":dataurl, "result_filename":result_filename}

        image_io = BytesIO()
        result_image_total.save(image_io, 'PNG')
        dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
        response = {"data_url":dataurl, "result_filename":result_filename}
        
        return HttpResponse(json.dumps(response), content_type="application/json")

    else:
        return HttpResponse("APIs for ShowOnMe are running!")

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