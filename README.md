
# JuNet
TensorFlow-based open source convolutional neural networks platform for medical image analysis

--------------------------------------------------------------------------------------------

의료 쪽에 종사하시는 여러 연구 교수님들 중 최근 열풍에 힘입어 딥러닝을 활용하고 싶어하시는 분들을 많이 만나 뵐 수 있었습니다. Fast Campus나 국가에서 운영하는 딥러닝 수업에 참여하시기도 하고 파이썬도 공부 하셨지만 직접 활용하지 못 한 채 그저 딥러닝에 특화되어있는 연구원들을 고용하려는 분들이 많았습니다. 하지만 가장 큰 문제는 딥러닝 연구원, 특히나 의료 분야의 딥러닝 연구원을 구하기가 하늘의 별따기 같다는 것 입니다. 그래서 그 분들에게 조금이라도 도움을 드리고자, 딥러닝과 파이썬의 기초를 이해하시는 분들이라면 쉽게 응용할 수 있도록 의료 전문 딥러닝 플랫폼을 만들어야겠다고 생각했습니다. 

국내에서 자율주행자동차나 로봇공학에서 연구하는 프로젝트들을 비롯하여 오픈된 관련 소스는 차고 넘칩니다. 그에 반해 의료 쪽 데이터는 제한된 환경을 기반으로 일방적 공개가 힘들기 때문에 정보 공유가 쉽지 않으며, 학습시키는 방법이 다른 데이터들과 크게 달라 어려움이 많습니다. 최소 이 플랫폼을 통해 의료 딥러닝 쪽에서 알려진 논문들이나 방법론들을 공유하고 의료 인공지능에 실질적으로 도입하는 비중이 높아졌으면 합니다. 이 플랫폼을 사용하면서 불편하다고 느끼시거나 추가 구현이 필요한 부분이 있다면 제게 말씀 부탁 드립니다.

--------------------------------------------------------------------------------------------

### 플랫폼 개발 목표
1. 의료 쪽 인공지능에 관심있는 사람이라면 누구든 이해하기 쉽게 제작.
2. 기본적인 의료 데이터가 세팅이 되는데로 바로바로 간단한 학습 및 테스트를 할 수 있도록 제작. 
3. 누구나 많이 알려진 의료 딥러닝 관련 논문들을 바로 적용 학습 할 수 있도록 구현. 
4. 의료 인공지능 쪽으로 일하는 많은 사람들이 참가 및 기본적인 노하우 공유 및 문서화.
5. 원하는 목표까지 개발이 이루어지면 더 나아가 해외까지 더 많은 사람들이 사용 하도록 유도. 

## Installation

```powershell
pip install git+https://gitlab.com/trackindatalabs/JuneNet.git
```

## 기본 구성

```python
import JuneNet

example_project = JuneNet.nn(project_title='example_project', work_dir='c://workspace/', data_ext='tfrecords',
                             num_gpus=None, tboard_record=True)

example_project.set_optimizer(subject_title='first_test',  # Title of the training subject. 
                              learning_rate=0.0001, decay_step=3, lr_decay_factor=1., num_epochs=100, batch_size=64, 
                              input_shape=None,  # Desired Shape of input image. None is default to input full image as it is.
                              inbound_shape=None,  # Limit the inbound size of the input image to set cropping image. 
                              grid_n=1,  # The number of cropping image in each data. 
                              valid_rate=0.1,  # Ratio of total data for Validation set 
                              model_name='vgg',  # Name of Networks that implemented. You can input Class typed customized network.
                              loss_name='ce',  # Name of Loss Function. 
                              optimizer='adam'  # Name of Optimization
                              )

example_project.train(report_per_epoch=100,  # Number of logs during one epoch. 
                      display_per_epoch=2, # Number of the result of Validaiotion Image during training
                      valid_per_epoch=50, # 
                      verbosity=2  # 1 = log status on tensorboard only, 2 = log status on  tensorboard and status
                      )
```

## Using Guide of JuNet

### raw 데이터로부터 tfrecords 생성 및 학습 준비하기. 

```python
org_data_dir = 'c://workspace/data/medical_image/'
image_list = glob(org_data_dir + '/*/*_ct.dcm')
label_list = glob(org_data_dir + '/*/*_label.dcm')
id_list = [data_path.split('medical_image\\')[-1].split('.')[0] for data_path in image_list]

JuneNet.data_manager.generate_dataset(project_title='dcm_example', id_list=id_list,
                                      image_list=image_list, label_list=label_list, name_classes=['benign', 'malignant'])
```

### 다양한 Hyper Parameter들을 한번에 돌려 비교 테스트하기. 

```python
example_project = JuneNet.nn("example_project")

for learning_rate in [0.001, 0.0001]:
    example_project.set_optimizer(subject_title='compare_lr',
                                  learning_rate=learning_rate)
    example_project.train('lr.%f' % learning_rate)
```
