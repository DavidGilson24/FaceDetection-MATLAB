clear all;
close all;

memory;
a = rand(20000);
memory;

% Webcam setup
faceCam = webcam();
faceCam.Resolution = '1280x720';

videoFrame = snapshot(faceCam);
videoPlayer = vision.VideoPlayer('Position', [100 100 520 520]);

faceDetect = vision.CascadeObjectDetector();
facePoints = vision.PointTracker('MaxBidirectionalError', 2);

% Variables initialization
isRunning = true;
numPoints = 0;
frameCount = 0;

% Audio recording setup
recorder = audiorecorder(44100, 16, 1);
isRecordingAudio = false;
silenceDuration = 0;
silenceThreshold = 0.02;

while isRunning
    
    videoFrame = snapshot(faceCam);
    grayScale = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    if numPoints >= 100
        markerColor = 'green';
    elseif numPoints >= 60
        markerColor = 'yellow';
    else
        markerColor = 'red';
    end
    
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
            videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 5, 'Color', 'white');
            videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', markerColor);
            titlePosition = [faceSquare(1), faceSquare(2) + faceSquare(4) + 10];
            videoFrame = insertText(videoFrame, titlePosition, 'Detected Face', 'FontSize', 16);
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
            videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 5, 'Color', 'white');
            videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', markerColor);
            
            titleY = max(facePolygon([2, 4, 6, 8])) + 10;
            titlePosition = [facePolygon(1), titleY];
            videoFrame = insertText(videoFrame, titlePosition, 'Detected Face', 'FontSize', 16);
            
            prevPoints = newPoints;
            setPoints(facePoints, prevPoints);
        end
    end
    
    % Audio recording logic
    if ~isRecordingAudio && numPoints > 0
        recordblocking(recorder, 2);
        isRecordingAudio = true;
        silenceDuration = 0;
    end

    if isRecordingAudio
        audioData = getaudiodata(recorder);
        if max(abs(audioData)) < silenceThreshold
            silenceDuration = silenceDuration + 1;
        else
            silenceDuration = 0;
        end
        
        if silenceDuration >= 2
            stop(recorder);
            isRecordingAudio = false;
            audioFileName = sprintf('voiceMessage_%s.wav', datestr(now, 'yyyymmdd_HHMMSS'));
            audiowrite(audioFileName, audioData, 44100);
        else
            recordblocking(recorder, 2);
        end
    end

    step(videoPlayer, videoFrame);
    if ~isOpen(videoPlayer)
        isRunning = false;
    end
end

if isRecording(recorder)
    stop(recorder);
end

% Cleanup
clear faceCam;
release(videoPlayer);
release(facePoints);
release(faceDetect);
clear recorder;


