# arcbook

## 학습

arcbook 프로젝트 상에서 해당 명령어로 실행 할 것
torchrun --nproc_per_node=2 train_v2.py configs/glint360k_r50.py

torchrun을 안할경우 gpu는 단일 gpu로 계산됨.

## 데이터 전처리
mxnet을 사용해야함으로 requirements.txt을 반드시 설치하고 mxnet도 설치해야함(mxnet은 gpu버전이어야함.)

```shell
# 형식은 반드시 이 형식을 따라야함.
/image_folder
├── 0_0_0000000
│   ├── 0_0.jpg
│   ├── 0_1.jpg
│   ├── 0_2.jpg
│   ├── 0_3.jpg
│   └── 0_4.jpg
├── 0_0_0000001
│   ├── 0_5.jpg
│   ├── 0_6.jpg
│   ├── 0_7.jpg
│   ├── 0_8.jpg
│   └── 0_9.jpg
├── 0_0_0000002
│   ├── 0_10.jpg
│   ├── 0_11.jpg
│   ├── 0_12.jpg
│   ├── 0_13.jpg
│   ├── 0_14.jpg
│   ├── 0_15.jpg
│   ├── 0_16.jpg
│   └── 0_17.jpg
├── 0_0_0000003
│   ├── 0_18.jpg
│   ├── 0_19.jpg
│   └── 0_20.jpg
├── 0_0_0000004

만약 한 폴더안에 전부 들어있다면 train 데이터와 val데이터를 나눌것

# train 데이터 lst, rec, idx 파일을 생성시키는 명령어
python -m mxnet.tools.im2rec --list --recursive train train
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train train
```

# val 데이터를 생성하는 방법
g_pair.py를 실행하고 나온 val_pair.txt를 val데이터 디렉토리에 반드시 넣을것.
넣은 뒤
lfw2pack.py를 실행하여 나온 val.bin을 train 디렉토리의 상위폴더에 넣을것

```shell
/arcbook_train_data
├── train
├── val.bin
├── val

이런식이 될것.
```
