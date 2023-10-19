clear all;
close all;

faceCam = webcam(); %initializing webcam object
faceCam.Resolution = '1280x720'; %setting resolution for the camera

videoFrame = snapshot(faceCam); %snapshot function used to capture every single frame
videoPlayer = vision.VideoPlayer('Position', [100 100 520 520]); % setting a frame for the video

faceDetect = vision.CascadeObjectDetector();

% Create an array to hold the point trackers for each face
faceTrackers = {};

%variables for the while loop
isRunning = true;
frameCount = 0; %setting how long the frame detector should work

while isRunning
    
    videoFrame = snapshot(faceCam);
    grayScale = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    faceSquares = faceDetect.step(grayScale);  % may detect multiple faces
    
    for i = 1:size(faceSquares, 1)  % iterate through each detected face
        faceSquare = faceSquares(i, :);
        if isempty(faceTrackers) || length(faceTrackers) < i  % if no tracker exists for this face, create one
            facePoints = vision.PointTracker('MaxBidirectionalError', 2);
            faceTrackers{end+1} = facePoints;
        else
            facePoints = faceTrackers{i};
        end
        
        points = detectMinEigenFeatures(grayScale, 'ROI', faceSquare);
        pointCoordinates = points.Location;
        numPoints = size(pointCoordinates, 1);
        
        % Initialize or re-initialize the point tracker whenever a new set of points is obtained
        release(facePoints);
        initialize(facePoints, pointCoordinates, grayScale);
        
        prevPoints = pointCoordinates;
        
                faceSquarePoints = bbox2points(faceSquare);
        
        if numPoints < 15
            facePolygon = reshape(faceSquarePoints', 1, []);
            videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', 'red');
        else
            [pointCoordinates, isFound] = step(facePoints, grayScale);
            newPoints = pointCoordinates(isFound, :);
            oldPoints = prevPoints(isFound, :);
            numPoints = size(newPoints, 1);
            
            if numPoints >= 15
                [xForm, oldPoints, newPoints] = estimateGeometricTransform(oldPoints, newPoints, 'similarity', 'MaxDistance', 4);
                faceSquarePoints = transformPointsForward(xForm, faceSquarePoints);  % Now faceSquarePoints is in scope
                facePolygon = reshape(faceSquarePoints', 1, []);
                videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 3);
                videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', 'green');
                prevPoints = newPoints;
                setPoints(facePoints, prevPoints);
            end
        end
        
        % Store the updated tracker back in the array
        faceTrackers{i} = facePoints;
    end
    
    step(videoPlayer, videoFrame);
    
    if ~isOpen(videoPlayer)
        isRunning = false;
    end
    
end

clear faceCam;
release(videoPlayer);
% Release all point trackers
for i = 1:length(faceTrackers)
    release(faceTrackers{i});
end
release(faceDetect);