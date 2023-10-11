%TEST 1

% clear all;
% close all;
% 
% faceDetector = vision.CascadeObjectDetector();
% 
% vid = videoinput('winvideo', 1);
% vid.ReturnedColorSpace = 'grayscale';
% vid.FramesPerTrigger = Inf;
% vid.FrameGrabInterval = 1;  % Grab every frame
% 
% videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);
% 
% start(vid);
% 
% while isrunning(vid)
%     frame = getdata(vid, 1, 'uint8'); % Get one frame at a time
%     
%     bbox = step(faceDetector, frame);
%     detectedImg = insertShape(frame, 'Rectangle', bbox, 'LineWidth', 5);
%     step(videoPlayer, detectedImg);
%     
%     if ~isOpen(videoPlayer)
%         break;
%     end
%     
%     flushdata(vid, 'triggers'); % Clear the data for the next iteration
% end
% 
% stop(vid);
% delete(vid);
% release(videoPlayer);

%TEST 2

clear all;
close all;

faceDetect = vision.CascadeObjectDetector();

vid = videoinput('winvideo', 1);
videoFrames = readFrame(vid);
faceBox = step(faceDetector, videoFrame);

videoFrame = insertShape(videoFrame, "rectangle", faceBox);
faceBoxArray = faceBox2points(faceBoxbox(1, :));



    
