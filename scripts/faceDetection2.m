clear all;
close all;

faceCam = webcam(); %initializing webcam object
faceCam.Resolution = '1280x720'; %setting resolution for the camera

videoFrame = snapshot(faceCam); %snapshot function used to capture every single frame
videoPlayer = vision.VideoPlayer('Position', [100 100 520 520]); % setting a frame for the video

faceDetect = vision.CascadeObjectDetector();
facePoints = vision.PointTracker('MaxBidirectionalError', 2); %will be used to track the face points

%variables for the while loop
isRunning = true;
numPoints = 0;
frameCount = 0; %setting how long the frame detector shoudl work

while isRunning
    
    videoFrame = snapshot(faceCam);
    grayScale = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    if numPoints < 15
        faceSquare = faceDetect.step(grayScale);
        
        if ~isempty(faceSquare)
            points = detectMinEigenFeatures(grayScale, 'ROI', faceSquare(1, :));
            
            pointCoordinates = points.Location;
            numPoints = size(pointCoordinates, 1);
            release(facePoints);
            initialize(facePoints, pointCoordinates, grayScale);
            
            prevPoints = pointCoordinates;
            
            square = bbox2points(faceSquare(1, :));
            facePolygon = reshape(square', 1, []);
            
            videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', 'green');
        end
        
    else
        [pointCoordinates, isFound] = step(facePoints, grayScale);
        newPoints = pointCoordinates(isFound, :);
        oldPoints = prevPoints(isFound, :);
        
        numPoints = size(newPoints, 1);
        
        if numPoints >= 15
            [xForm, oldPoints, newPoints] = estimateGeometricTransform(oldPoints, newPoints, 'similarity', 'MaxDistance', 4);
            
            square = transformPointsForward(xForm, square);
            facePolygon = reshape(square', 1, []);
            
            videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', 'green');
            
            prevPoints = newPoints;
            setPoints(facePoints, prevPoints);
        end
    end
    
   step(videoPlayer, videoFrame);
   
   if ~isOpen(videoPlayer)
       isRunning = false;
   end
   
end

clear faceCam;
release(videoPlayer);
release(facePoints);
release(faceDetect);



        
        
