# vit embedding server

a c++ server that provides vision transformer (vit) embeddings via http api using libtorch and crow. this is essentially two projects  [vit_export_experiment](https://github.com/johngunerli/ViT_cpp_experiment) and [embedding python and cpp server](https://github.com/johngunerli/embedding-server-cpp).

## prerequisites

- cmake (3.0+)
- libtorch
- crow (header-only library)
- c++17 compatible compiler

## project structure

```bash
.
├── vit_embedding.pt          # trained vit model (needs to be jit.traced from huggingface)
├── vit_embedding_server/     # server implementation
│   ├── CMakeLists.txt
│   └── main.cpp
└── libtorch/                 # libtorch dependencies
```

## build instructions

```bash
cd vit_embedding_server
mkdir build && cd build
cmake ..
make
```

## usage

1. start the server:

```bash
./vit_embedding_server
```

2. the server exposes two endpoints:
   - `GET /`: returns basic usage information
   - `POST /embed`: accepts image data and returns embedding

## testing

you can test the server using the following python script:

````python
import json
import numpy as np
import requests
from PIL import Image

SERVER_URL = "http://localhost:8080/embed"
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  
    image = np.array(image)
    image_list = image.tolist()  
    json_payload = json.dumps({"image": image_list})

    return json_payload

def send_request(json_payload):
    headers = {"Content-Type": "application/json"}
    response = requests.post(SERVER_URL, data=json_payload, headers=headers)

    if response.status_code == 200:
        print("✅ Server Response:", response.text)
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    image_path = "./Edgar_Degas_-_La_Classe_de_danse.jpg"
    json_payload = preprocess_image(image_path)
    send_request(json_payload)

````

## testing result

for the image ![](./Edgar_Degas_-_La_Classe_de_danse.jpg)

the response we get is:

````bash
$python test.py 
✅ Server Response: Columns 1 to 10 0.0279 -0.1036  0.3922  0.5302 -0.2285 -0.8940 -0.3910 -0.4888 -0.2387 -0.6078

Columns 11 to 20-0.0857 -0.0280 -0.3191 -0.2831  0.0523  0.0588 -0.4169 -0.0907  0.3657  0.3709

Columns 21 to 30-0.4007  0.1461  0.6338 -1.3776  0.6975 -0.7119 -0.6276 -0.3358 -0.6161 -0.6522

Columns 31 to 40 0.5434 -0.8713 -0.5905 -2.0933  0.9809 -0.2983  0.1387  0.0140  0.2760  0.1505

Columns 41 to 50-0.7077  1.1579  0.9285 -0.0835  0.2940 -0.6386  0.2807 -0.9463 -0.3970  0.4724

Columns 51 to 60 0.9670  0.4437  0.3644  0.7874 -0.2027 -0.4203  0.6352  0.5903 -0.8425  0.1040

Columns 61 to 70 0.3149 -0.1219 -0.9111  0.8346  0.1068  0.3179 -0.6609 -0.7252 -0.1579 -0.0351

Columns 71 to 80-0.6699  0.1735 -0.0383 -0.1448  0.1649  0.2251  0.5000 -0.7264 -0.0647  0.1052

Columns 81 to 90 0.1578  0.4336 -0.0544 -0.9277 -0.6100 -0.6678 -0.4738  0.1491 -0.6392 -0.0523

Columns 91 to 100 0.2572 -0.8529  0.4938  0.5937  0.2132  0.0339 -0.1300 -0.5276 -0.2911  0.2030

Columns 101 to 110-0.6762  0.1547  0.0591 -1.1158  0.6360  0.0686 -0.2991 -0.2674  1.1495 -0.0952

Columns 111 to 120-0.0359  0.2676  0.9508  0.5632 -0.2585 -0.1762 -0.4338  0.0888 -0.4328  0.0409

Columns 121 to 130-0.1884  0.4014 -1.2548 -0.1583  0.2511  0.4918  0.0467  0.4012 -0.1252  0.5583

Columns 131 to 140-1.4606 -0.1081 -0.2346  0.3412  0.0838  0.1254 -0.8419  0.0733 -0.0292 -0.3358

Columns 141 to 150-0.4192  0.0747  0.7356 -0.4836  0.6582 -1.1187  0.3050 -0.9961  0.0232  0.2327

Columns 151 to 160-0.0102  0.4178 -0.1721 -0.4795 -1.3231 -0.6303 -0.6020  0.3570 -0.2594 -0.2051

Columns 161 to 170-0.2424  0.2545  0.4109  0.6490  0.1641 -0.0746  0.3185  0.9195 -0.7359  0.8069

Columns 171 to 180-0.3721 -0.9541 -0.1702 -0.6513  0.2247  0.0352  0.7819 -0.4025  0.0366  0.3559

Columns 181 to 190-0.8809 -0.2039 -0.2837  0.1989  0.5436 -0.8942  0.8408 -5.9719  0.0421  0.8262

Columns 191 to 200-0.3465 -0.4686  0.1979  0.5034 -0.4094  0.1671  0.1839  0.6850  0.2299  0.8292

Columns 201 to 210 0.8165 -0.3078  1.3067  0.7793  0.1709  0.8859  0.1293  0.0562 -0.3489 -0.8223

Columns 211 to 220 1.4150 -0.4843  0.7680  0.1668  0.0684 -0.4246 -0.4475  0.0415 -0.3873 -0.8313

Columns 221 to 230-0.9700 -0.7085  0.4453  0.2049  0.8351 -0.1485 -0.1670  0.0701  0.1200 -0.7982

Columns 231 to 240-0.3136  0.0241 -0.9758 -0.4099  0.7451 -0.6885 -0.7397 -0.2312 -0.7326 -0.7163

Columns 241 to 250 1.1959  0.2594  0.8500 -0.5433  0.7057  0.4147  0.5524 -0.0825  0.9267  0.0144

Columns 251 to 260 0.1386 -0.8006  0.3132  0.3421 -0.0918 -0.5736 -0.1793 -0.1837 -0.1362 -1.0435

Columns 261 to 270-0.0996  0.2149 -0.1955  0.3954  0.5149  0.1960 -0.7906 -0.4520  0.0361  0.6648

Columns 271 to 280-0.4850  0.6045  0.7251  0.0559 -0.0554  0.7834  0.8000  0.2006 -0.2609 -0.4044

Columns 281 to 290-0.6267 -0.7683  0.0984 -0.0645 -0.1688 -0.3202 -1.1133  0.6379  0.7418  1.4032

Columns 291 to 300 0.0240  0.3254  0.0787 -0.7076  0.3192 -0.4626  0.1053 -0.2953 -0.3397 -0.8766

Columns 301 to 310 0.0729 -0.1836 -0.8158 -0.2008  0.1561 -0.4600 -0.3606  0.7935 -0.8166  0.1034

Columns 311 to 320 0.2795  1.0889  0.6447 -0.8775 -1.0793  0.0855 -0.0614 -0.4162 -0.2563 -0.4459

Columns 321 to 330 0.2352 -0.1362  0.2054 -0.0230 -0.5435 -0.7621  0.3851 -0.1353 -0.1777 -0.6217

Columns 331 to 340-0.2370 -0.4749  0.9590  0.9287 -0.3578 -0.5415  0.1804 -0.0706 -0.0634 -0.0117

Columns 341 to 350-0.4571 -0.3799 -0.6057 -0.1762 -1.3293 -0.2244 -0.5977 -0.9027 -0.2197 -0.6725

Columns 351 to 360 0.6435 -0.8366 -0.0362  0.3693 -0.1400 -0.4098  0.0435  0.3510  0.5389  0.2766

Columns 361 to 370-1.6334 -0.5980 -0.3656 -1.0033 -0.0958 -0.2648  0.4900  0.4046  0.6090 -0.1302

Columns 371 to 380-0.1368  0.5479  0.0358 -1.0813  1.1959  1.5039 -0.5785  0.2788  1.1973 -0.6005

Columns 381 to 390-0.2009 -0.9862 -0.1179 -0.0609 -0.6323 -0.6143  0.4472  0.7076 -0.1338  0.0453

Columns 391 to 400 0.4178 -0.1901  0.0431  0.2056 -0.1291  0.4765  0.4861  0.0615  0.3225  0.2493

Columns 401 to 410-0.3391 -0.4965  0.0302 -0.2707 -0.8301 -0.5587 -0.2948 -0.7488 -0.0460  1.6445

Columns 411 to 420-0.3358  0.3350  0.3145  0.0821 -0.7379  0.7085 -1.4276 -0.3061 -0.2935  0.4035

Columns 421 to 430 0.2061 -0.1389 -0.1898  0.9363 -0.0624  0.2046  0.0916 -0.0078  0.8699  0.1569

Columns 431 to 440-0.3027  0.0478  0.6097  0.0759  0.3702 -0.1746  0.4216 -0.0620  0.4485  0.3643

Columns 441 to 450-0.1088  0.0387 -0.8267  0.8966  0.1483  0.0899 -0.4125  0.2274 -0.0516  0.5032

Columns 451 to 460-0.0219  0.5230  0.4211 -0.2076 -0.4604  0.1154  1.0022  0.1127  0.9702  0.6683

Columns 461 to 470-0.3884  0.7709  0.0708 -0.4860 -0.7612 -1.3429 -0.8243 -0.4229 -0.1882 -0.2151

Columns 471 to 480-0.1701 -0.6774  0.3272 -0.3139  0.4561 -0.0036 -0.8706  0.4666  0.3601 -0.9196

Columns 481 to 490 0.2323  0.2497  0.8130 -0.2154 -0.1098 -0.5238  0.2225  1.0702  0.1560  0.0520

Columns 491 to 500-1.1389 -0.8874  0.3158 -0.1145  0.9128 -1.0935  0.7648 -0.1086 -0.3909  0.2477

Columns 501 to 510 0.5529  0.2675  0.1279  0.2071 -0.3746  0.2634 -0.4129  0.0216  0.1289 -0.5952

Columns 511 to 520-0.4365  0.0591  0.9777 -0.3298 -0.3845  0.0270  0.7804 -0.6178 -0.1019 -0.5600

Columns 521 to 530 0.2258 -0.3027 -0.1928  0.1353  0.3996 -0.2291 -0.3523 -0.8435 -0.1198  0.0924

Columns 531 to 540-0.4309 -0.2041  0.0865  0.5038  0.6425 -2.4003 -0.0984  1.0654 -0.7346 -0.0088

Columns 541 to 550-0.0761 -0.1123 -0.3407  0.5395  0.6223  0.9195  0.4220  1.2213  0.3816  0.7753

Columns 551 to 560-0.2354  0.7049  0.3976 -0.7187  0.4270  0.5156  0.4891  1.0375 -0.4377 -0.3065

Columns 561 to 570 0.5190 -0.3755  0.4721  0.8339  0.8944  0.9380  0.7388  0.9077  0.4336  0.4927

Columns 571 to 580-0.0533 -0.7540 -0.3759  0.2042 -0.2554 -0.4758 -0.5930 -0.1609  0.0178 -0.7108

Columns 581 to 590-0.3599  0.7251 -0.8356  0.1478  0.3251  0.2006 -0.3677 -0.8185  0.8112  0.3066

Columns 591 to 600-0.4899 -0.5590 -0.0635  0.0970  1.1305  0.5552  0.1449 -0.5666 -0.4208 -0.8535

Columns 601 to 610 1.5627 -0.7154 -0.1334 -0.6014  0.3096  0.3995 -0.1885 -0.3055 -0.6946 -0.0932

Columns 611 to 620-0.1161  0.4919 -0.2456  0.9335  0.1175  0.4788  0.4420  0.5950  0.8175  0.1709

Columns 621 to 630 1.2404 -0.8426  0.3327 -0.6970 -0.3407 -0.9138 -1.1269  0.5014  0.4626  0.1620

Columns 631 to 640-0.5267  0.6424 -0.6130 -0.7835 -0.0520 -1.3024 -0.0926 -0.9870  0.3471 -0.4183

Columns 641 to 650-0.0390 -0.8280 -0.3675 -0.3284  0.4429 -0.2309  0.3127  1.2784  0.1406 -0.2373

Columns 651 to 660 0.3552 -0.5996  0.4562 -0.2633 -0.5769  0.4763  0.7687  1.2026 -1.1260 -0.2409

Columns 661 to 670 0.4421  0.5042 -0.7002  1.3828  0.1700  0.7354 -0.2325  0.0192  0.8197  0.9191

Columns 671 to 680-0.9856  0.8150  1.3647 -0.7907  0.2273 -0.5338 -0.5241  0.3524 -0.1579  0.1199

Columns 681 to 690 0.2142  0.6235  0.2403  0.6231  0.2320  0.5832 -0.0998 -0.9350 -0.4344  0.3871

Columns 691 to 700-0.5068  0.6099 -0.3087 -0.3502  0.3067  0.4863 -0.3845  0.1400  0.2546  0.4182

Columns 701 to 710-0.6644  0.2698  0.4133 -0.3254 -0.8424  0.8026  0.1419  0.3972 -0.2411 -0.2444

Columns 711 to 720-0.3206 -0.5390 -0.2482 -0.5847  0.7851  0.3568  0.7879  0.1835  0.1547 -1.3142

Columns 721 to 730-0.2973  0.0326  0.0286 -0.0457 -0.2945  0.3407  0.6664 -0.2956  0.7391  0.7203

Columns 731 to 740 0.0587  0.2521  0.2947  0.3868  0.8221  0.2773 -0.1116  0.1079  0.3258  1.1874

Columns 741 to 750 0.0921  0.0241 -0.7512 -0.4604 -0.4159  0.2420  0.3076  1.2374  0.6351 -0.3486

Columns 751 to 760-0.4452  0.1818 -0.5546  0.0415 -0.0939 -0.0929 -0.0715 -0.6826 -0.8936 -0.1390

Columns 761 to 768 2.1753  0.4586 -0.3533 -0.0035  0.2631 -0.2785  0.8714 -0.6045
[ CPUFloatType{1,768} ]
````
