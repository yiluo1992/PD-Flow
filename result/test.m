img1_rgb = imread('i1.png');
img2_rgb = imread('i2.png');

img1_gray = rgb2gray(img1_rgb);
img2_gray = rgb2gray(img2_rgb);

img1_gray_resized = imresize(img1_gray, 0.5);
img2_gray_resized = imresize(img2_gray, 0.5);

load pdflow_results02.txt;

flow_u = pdflow_results02(:,4);
flow_v = pdflow_results02(:,5);
flow_w = pdflow_results02(:,6);

img_u = reshape(flow_u, 320, 240)';
img_v = reshape(flow_v, 320, 240)';
img_w = reshape(flow_w, 320, 240)';

img_warped = zeros(240,320);

for r = 1 : 240
    for c = 1 : 320
        
%         if(img_w(r,c) ~= 0)
%             c_new = round(c + img_u(r,c));
%             r_new = round(r + img_v(r,c));
%             img_warped(r,c) = img2_gray_resized(r_new, c_new);
%         else
%             img_warped(r,c) = img2_gray_resized(r, c);
%         end

c_new = round(c + img_u(r,c));
r_new = round(r + img_v(r,c));
if(c_new < 1 || c_new > 320)
    c_new = c;
end
if(r_new < 1 || r_new > 240)
    r_new = r;
end
img_warped(r,c) = img2_gray_resized(r_new, c_new);

    end
end

img_warped = uint8(img_warped);

figure;
imshow(img1_gray_resized-img2_gray_resized);

figure;
imshow(img1_gray_resized-img_warped);