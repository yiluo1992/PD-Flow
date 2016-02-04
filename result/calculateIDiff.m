img1_rgb = imread('i1.png');
img2_rgb = imread('i2.png');

img1_gray = rgb2gray(img1_rgb);
img2_gray = rgb2gray(img2_rgb);

img1_gray_resized = imresize(img1_gray, 0.5);
img2_gray_resized = imresize(img2_gray, 0.5);

load pdflow_results01.txt;

flow_u = pdflow_results01(:,4);
flow_v = pdflow_results01(:,5);
flow_w = pdflow_results01(:,6);

img_u = reshape(flow_u, 320, 240)';
img_v = reshape(flow_v, 320, 240)';
img_w = reshape(flow_w, 320, 240)';

img_warped = zeros(240,320);
for r = 1 : 240
    for c = 1 : 320  
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

% interpolate
img_warped_interpolate = zeros(240,320);
for r = 1 : 240
    for c = 1 : 320  
        cc = (c + img_u(r,c));
        rr = (r + img_v(r,c));
        
        if(cc < 1 || cc > 320)
            cc = c;
        end
        if(rr < 1 || rr > 240)
            rr = r;
        end
              
        c1 = floor(cc);
        c2 = c1+1;
        r1 = floor(rr);
        r2 = r1+1;
        
        if(c2 > 320)
            c2 = c1;
        end
        if(r2 > 240)
            r2 = r1;
        end
        
        part1 = (c2-cc)*(img2_gray_resized(r1,c1)) + (cc-c1)*(img2_gray_resized(r1,c2));
        part2 = (c2-cc)*(img2_gray_resized(r2,c1)) + (cc-c1)*(img2_gray_resized(r2,c2));
        val_interpolate = (r2-rr)*part1 + (rr-r1)*part2;
  
        img_warped_interpolate(r,c) = val_interpolate;
        
    end
end
img_warped_interpolate = uint8(img_warped_interpolate);



IniIDiff = img1_gray_resized-img2_gray_resized;
IDiff = img1_gray_resized-img_warped;
IDiff_interpolate = img1_gray_resized-img_warped_interpolate;
IDiff_interpolate(240,:) = IDiff_interpolate(240,:)*0;
IDiff_interpolate(:,320) = IDiff_interpolate(:,320)*0;

figure;
imshow(IniIDiff);

figure;
imshow(IDiff);

figure;
imshow(IDiff_interpolate);

imwrite(IniIDiff, './iniIDiff.png');
imwrite(IDiff, './resultIDiff.png');
imwrite(IDiff_interpolate, './resultIDiff_interpolate.png');