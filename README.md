# Emojinator
a deep learning model that recognizes and classify different hand emojis

### DEMO
click to watch!

[![Watch the video](https://img.youtube.com/vi/SpVEFF7TnXU/maxresdefault.jpg)](https://youtu.be/SpVEFF7TnXU)

### To make the gesture dataset:
 ```
make sure to create a gesture folder wherever you want, and change the path in the "emoji.py"
wherever necessary.
 
 run:  emoji.py
  
When you see the frame and thresh screen , press "c" and show your hand gesture(only 1 gesture) 
in front of webcam.
  - it will start capturing until count 1200 (you can see on screen) & the program will quit.
  - continue the above steps for how many ever gestures you want.(in my code 11 gestures)

this code will create a gesture dataset with different hand gestures. 
  ```
  
### convert gesture folder to csv for training the model:
 ```
 run:  createCSV.py
this code will create a csv file named train_foo.csv. 
  ```
### training the model:
 ```
 run:  trainEmoji.py
this code will train the model and create the weight file emojinator.h5. 
  ``` 
### Emojinator:
 ```
 run:  emojinator.py
this code will predict your hand gestures according to the dataset provided. 
  ```   
