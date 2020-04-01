# HumanActivityRecognition
This moduel will include two things ( Human activity Recognition + face identification ) in real time

After cloning this repo
download a video and save that video in the current working directory with the file name as example_activities.mp4 

For weight file download resnet-34_kinetics.onnx (The resnet weights on the Kinetics dataset).

To upload larger file to your repo
use : git lfs track "*.mp4"

      git lfs track "*.onnx"

Now make sure .gitattributes is tracked:

      git add .gitattributes
      
then normally proceed with 
      git add .
      
      git commit -m "message"
      
      git push
