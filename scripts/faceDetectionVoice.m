close all;

% video setup
faceCam = webcam(); % initializing webcam object
faceCam.Resolution = '640x360'; % setting resolution for the camera

videoFrame = snapshot(faceCam); 
videoPlayer = vision.VideoPlayer('Position', [100 100 520 520]); % setting a frame for the video

faceDetect = vision.CascadeObjectDetector(); % object detection using the viola jones algorithm
facePoints = vision.PointTracker('MaxBidirectionalError', 50); % used to track the face points

% Variables for the while loop
isRunning = true;
numPoints = 0;
frameCount = 0; % setting how long the frame detector should work (only used for timed tests and uses)

prevFaceDetected = false;

% recording setup
recorder = audiorecorder(44100, 16, 1);
isRecordingAudio = false;
silenceThreshold = 0.01; % used to check how loud the audio is being picked up
checkDuration = 0.5; % duration between checks for silence
silenceWindowSize = 3; % duration of silence to trigger end of recording

while isRunning
    
    videoFrame = snapshot(faceCam);
    grayScale = rgb2gray(videoFrame); %convert o grayscale since point detection works best in grayscale
    frameCount = frameCount + 1;
    
    % color based point detection to let the user know if the tracking is
    % deprecated
    if numPoints >= 100
        markerColor = 'green'; % green if tracking is not close to being reset
    elseif numPoints >= 60
        markerColor = 'yellow'; % yellow if tracking has lost some points
    else
        markerColor = 'red'; % red if tracking loses lots of points and will reset
    end
    
    if numPoints < 30 % if the current frame has less than 30 points then:
        faceSquare = faceDetect.step(grayScale); % detect face in a grayscale frame
        
        if ~isempty(faceSquare) % checks if face is detected
            points = detectMinEigenFeatures(grayScale, 'ROI', faceSquare(1, :)); % detect corner points on face

            % tracking the points with coordinates and count them
            pointCoordinates = points.Location; 
            numPoints = size(pointCoordinates, 1);
            release(facePoints);
            initialize(facePoints, pointCoordinates, grayScale);

            prevPoints = pointCoordinates;

            % converting polygon points to face square
            square = bbox2points(faceSquare(1, :));
            facePolygon = reshape(square', 1, []);
            
            % add square around face
            videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 5, 'Color', 'white');
            videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', markerColor);
            titlePosition = [faceSquare(1), faceSquare(2) + faceSquare(4) + 10]; % add title under face
            videoFrame = insertText(videoFrame, titlePosition, 'Detected Face', 'FontSize', 16);
        end
        
    else
        [pointCoordinates, isFound] = step(facePoints, grayScale); % track the points in the square
        newPoints = pointCoordinates(isFound, :); % filter points that were tracked in the current frame

        % get previous corresponding points that were tracked in the current frame and update their number
        oldPoints = prevPoints(isFound, :);
        numPoints = size(newPoints, 1);
        
        if numPoints >= 30
            % estimate the movement between old and new points
            % check used to see how the face  moved between frames using the points
            [xForm, oldPoints, newPoints] = estimateGeometricTransform(oldPoints, newPoints, 'similarity', 'MaxDistance', 4);
            
            %move square w/ that estimated movement
            square = transformPointsForward(xForm, square);
            facePolygon = reshape(square', 1, []);
            
            videoFrame = insertShape(videoFrame, 'Polygon', facePolygon, 'LineWidth', 5, 'Color', 'white'); % draw square arround the face
            videoFrame = insertMarker(videoFrame, pointCoordinates, 'x', 'Color', markerColor); % draw points on face

            %placement for the title
            titleY = max(facePolygon([2, 4, 6, 8])) + 10;
            titlePosition = [facePolygon(1), titleY];
            videoFrame = insertText(videoFrame, titlePosition, 'Detected Face', 'FontSize', 16); % draw title under square

            % changing points for tracking in the next frame
            prevPoints = newPoints;
            setPoints(facePoints, prevPoints);
        end

    end

    faceDetected = numPoints > 0; % check for face detection

    if faceDetected && ~prevFaceDetected && ~isRecordingAudio % start the recording if face is in frame
        record(recorder);
        isRecordingAudio = true;
        
    elseif ~faceDetected && isRecordingAudio % stop recording if face is no longer in frame
        stop(recorder);
        audioFileName = sprintf('voiceMessage_%s.wav', datestr(now, 'yyyymmdd_HHMMSS')); % setting name for audio file
        audiowrite(audioFileName, getaudiodata(recorder, 'double'), recorder.SampleRate);
        isRecordingAudio = false;
    end

    if isRecordingAudio
        pause(checkDuration);
        
        audioData = getaudiodata(recorder, 'double');
        
        % silence check
        if length(audioData) > silenceWindowSize * recorder.SampleRate
            segmentData = audioData(end-round(recorder.SampleRate*silenceWindowSize):end);
            
            if max(abs(segmentData)) < silenceThreshold % checks the volume threshold
                stop(recorder);
                fprintf('Stopped recording due to silence.\n');
                audioFileName = sprintf('voiceMessage_%s.wav', datestr(now, 'yyyymmdd_HHMMSS'));
                audiowrite(audioFileName, audioData, recorder.SampleRate);
                isRecordingAudio = false;
            end
        end
    end

    prevFaceDetected = faceDetected;

    step(videoPlayer, videoFrame); % show video frame.
    
    if ~isOpen(videoPlayer) % stop program if the video player is closed
        isRunning = false;
    end
end

% clean workspace
clear faceCam;
release(videoPlayer); 
release(facePoints);
release(faceDetect);



