faceDetector = vision.CascadeObjectDetector();

vid = videoinput('winvideo', 1);
vid.ReturnedColorSpace = 'grayscale';
vid.FramesPerTrigger = Inf;
triggerconfig(vid, 'manual');
start(vid);

trigger(vid);
videoFrame = getdata(vid);

bbox = step(faceDetector, videoFrame);

videoFrame = insertShape(videoFrame, "rectangle", bbox);
figure; imshow(videoFrame); title("Detected face");

bboxPoints = bbox2points(bbox(1, :));

points = detectMinEigenFeatures(im2gray(videoFrame), "ROI", bbox);

figure, imshow(videoFrame), hold on, title("Detected features");
plot(points);

pointTracker = vision.PointTracker("MaxBidirectionalError", 2);

points = points.Location;
initialize(pointTracker, points, videoFrame);

videoPlayer  = vision.VideoPlayer("Position",...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

oldPoints = points;

% Use a loop to continuously capture frames from the webcam
while ishandle(videoPlayer.Parent)
    trigger(vid);
    videoFrame = getdata(vid);

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        [xform, inlierIdx] = estimateGeometricTransform2D(...
            oldInliers, visiblePoints, "similarity", "MaxDistance", 4);
        oldInliers    = oldInliers(inlierIdx, :);
        visiblePoints = visiblePoints(inlierIdx, :);
        
        bboxPoints = transformPointsForward(xform, bboxPoints);
        bboxPolygon = reshape(bboxPoints', 1, []);
        videoFrame = insertShape(videoFrame, "polygon", bboxPolygon, "LineWidth", 2);
        videoFrame = insertMarker(videoFrame, visiblePoints, "+", "MarkerColor", "white");       
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);        
    end
    
    step(videoPlayer, videoFrame);
end

% Clean up
stop(vid);
delete(vid);
release(videoPlayer);


    
