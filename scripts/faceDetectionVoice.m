close all;

% video setup
faceCam = webcam(); % initializing webcam object
faceCam.Resolution = '640x360'; % setting resolution for the camera

videoFrame = snapshot(faceCam); 
videoPlayer = vision.VideoPlayer('Position', [100 100 520 520]); % setting a frame for the video

faceDetect = vision.CascadeObjectDetector();
facePoints = vision.PointTracker('MaxBidirectionalError', 50); % used to track the face points

% Variables for the while loop
isRunning = true;
numPoints = 0;
frameCount = 0; % setting how long the frame detector should work

prevFaceDetected = false; % Add this line

% recording setup
recorder = audiorecorder(44100, 16, 1);
isRecordingAudio = false;
silenceThreshold = 0.01;
checkDuration = 0.5; % duration between checks for silence
silenceWindowSize = 3; % duration of silence to trigger end of recording

while isRunning
    
    videoFrame = snapshot(faceCam);
    grayScale = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    % color based point detection to let the user know if the tracking is
    % deprecated
    if numPoints >= 100
        markerColor = 'green';
    elseif numPoints >= 60
        markerColor = 'yellow';
    else
        markerColor = 'red';
    end
    
    if numPoints < 30
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
        
        if numPoints >= 30
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
    
    % fprintf('Number of detected points: %d\n', numPoints);
    
    faceDetected = numPoints > 0;

    if faceDetected && ~prevFaceDetected && ~isRecordingAudio
        % fprintf('Started recording audio...\n');
        record(recorder); % start continuous recording
        isRecordingAudio = true;
    elseif ~faceDetected && isRecordingAudio
        % Face is no longer detected but recording is active
        % fprintf('Face is no longer detected. Stopping recording.\n');
        stop(recorder);
        audioFileName = sprintf('voiceMessage_%s.wav', datestr(now, 'yyyymmdd_HHMMSS'));
        audiowrite(audioFileName, getaudiodata(recorder, 'double'), recorder.SampleRate);
        isRecordingAudio = false;
    end

    if isRecordingAudio
        pause(checkDuration); % pause for a duration before checking for silence
        
        audioData = getaudiodata(recorder, 'double');
        
        if length(audioData) > silenceWindowSize * recorder.SampleRate
            segmentData = audioData(end-round(recorder.SampleRate*silenceWindowSize):end);
            
            if max(abs(segmentData)) < silenceThreshold
                % Silence detected, stopping the recording
                stop(recorder);
                fprintf('Stopped recording due to silence.\n');
                audioFileName = sprintf('voiceMessage_%s.wav', datestr(now, 'yyyymmdd_HHMMSS'));
                audiowrite(audioFileName, audioData, recorder.SampleRate);
                isRecordingAudio = false;
            end
        end
    end

    prevFaceDetected = faceDetected;

    step(videoPlayer, videoFrame);
    
    if ~isOpen(videoPlayer)
        isRunning = false;
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



