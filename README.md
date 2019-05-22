# Open-Source S/W project - team 11

Besaic
====
Besaic는 실시간 스트리밍 환경에서 사람들의 얼굴을 자동으로 모자이크 처리 해줍니다. 또한 방송에 부적합한 음란한 내용을 NSFW(Not suitable for work)셋을 이용해 모자이크 합니다.

Reference
----
Face recognition : https://github.com/ageitgey/face_recognition
Face tracking : https://github.com/gdiepen/face-recognition
NSFW training data :  https://github.com/alexkimxyz/nsfw_data_scraper
NSFW pre-trained model :  https://github.com/GantMan/nsfw_model

얼굴 모자이크
----

설명 솰라솰라

NSFW
----
# NSFW Detection Machine Learning Model
Trained on 60+ Gigs of data to identify:
- `drawings` - safe for work drawings (including anime)
- `hentai` - hentai and pornographic drawings
- `neutral` - safe for work neutral images
- `porn` - pornographic images, sexual acts
- `sexy` - sexually explicit images, not pornography

This model powers [NSFW JS](https://github.com/infinitered/nsfwjs) - [More Info](https://shift.infinite.red/avoid-nightmares-nsfw-js-ab7b176978b1)

### 속도
모델의 크기가 커지면 정확도가 높아지긴 하지만 그만큼 계산량 또한 많아져 속도가 떨어집니다. 그래서 적당한 정확도를 갖는 모델 중에서 크기가 작아 속도가 빠른 모델을 base 모델로 선정하였습니다. F-Score 값이 95 이상이면서 모델의 크기가 작은 모델은 win=3, emb=30이며 F-Score는 95.30입니다.

속도를 비교하기 위해 1만 문장(총 903KB, 문장 평균 91)의 텍스트를 분석해 비교했습니다. base 모델의 경우 약 10.5초, large 모델의 경우 약 78.8초가 걸립니다.

빌드 및 설치
----
```console
pip install cmake dlib opencv-python face_recognition numpy
```

#  Minimum UI

1. 빌드를 위해 필요한 것들
* PyQT5
* Python 3. over
2. Code preview
```python
layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)
        self.btn = QPushButton("NSFW")
        layout.addWidget(self.btn)
        self.le = QLabel("")

        layout.addWidget(self.le)
        self.btn.clicked.connect(self.getfile)
        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setLayout(layout)
```
* start button 은 모자이크 처리를 시작을 뜻함
* NSFW button 은 이미지를 받아와서 NSFW 필터링을

## Usage
```python
from nsfw_detector import NSFWDetector
detector = NSFWDetector('./nsfw.299x299.h5')
```

##Demo
- https://www.youtube.com/watch?v=_IUhD4zoYuI


Contributing
----
Besaic에 기여하실 분들은
