import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

os.chdir("C:/Users/Admin/Desktop")

imgx = cv2.imread("elephant.jpg")
imgy = cv2.imread("tiger.jpg")
 
gray_image = cv2.cvtColor(imgx, cv2.COLOR_BGR2GRAY)

#Task1
def hog(img, number_of_directions, number_of_grid_dimension):
    # Calculate the gradient of the image in x and y direction
    gradx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    
    global gradient_magnitude
    # Calculate the gradient magnitude and direction
    gradient_magnitude = np.sqrt(np.square(gradx) + np.square(grady))
   
    gradient_direction = np.arctan2(grady, gradx)
    
    # Divide the image into cells with given grid dimension
    cell_r1, cell_c1 = number_of_grid_dimension
    cell_hgt, cell_wdth = img.shape[0]//cell_r1, img.shape[1]//cell_c1
    
    histogram = np.zeros((cell_r1, cell_c1, number_of_directions))
    
    for row in range(cell_r1):
        for col in range(cell_c1):
            # Get the gradient information for each cell
            gradient_magnitudecell = gradient_magnitude[row*cell_hgt:(row+1)*cell_hgt, col*cell_wdth:(col+1)*cell_wdth]
            gradient_directioncell = gradient_direction[row*cell_hgt:(row+1)*cell_hgt, col*cell_wdth:(col+1)*cell_wdth]
            
            # Calculate the histogram of oriented gradients for each cell
            bin_size = np.pi / number_of_directions
            for i in range(number_of_directions):
                bin_start = i * bin_size
                bin_end = bin_start + bin_size
                indices = np.where((gradient_directioncell >= bin_start) & (gradient_directioncell < bin_end))
                histogram[row, col, i] = np.sum(gradient_magnitudecell[indices])
    hist=str(histogram.ravel())
    print("Histogram: " + hist+"\n"+"\n"+"\n")
 
    return histogram.ravel()
    
#hog(imgx,9,(8,8))
 

#Task2
def hog_similarity(imgx, imgy, number_of_directions, number_of_grid_dimension):
    hog1 = hog(imgx, number_of_directions, number_of_grid_dimension) #calculate histogram of image1 using hog() function

    hog2 = hog(imgy, number_of_directions, number_of_grid_dimension) #calculate histogram of image2 using hog() function
    
    # Calculating the cosine similarity
    product = np.dot(hog1, hog2)
    magn1 = np.sqrt(np.sum(np.square(hog1)))
    magn2 = np.sqrt(np.sum(np.square(hog2)))
    cosineSimilarity = product / (magn1 * magn2)
    cos_sim=str(cosineSimilarity)
    print("Cosine similarity: "+cos_sim+ "\n")

hog_similarity(imgx,imgy,9,(8,8))





def hog_occurances(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize the images to a common size
    gray1 = cv2.resize(gray1, (500, 500))
    gray2 = cv2.resize(gray2, (500, 500))

    # Compute the histograms for the grayscale images
    histx = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    histy = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize the histograms
    histx = cv2.normalize(histx, histx).flatten()
    histy = cv2.normalize(histy, histy).flatten()

    # Compute the cosine similarity between the histograms
    simil = cosine_similarity([histx], [histy])[0][0]

    # Threshold the similarity score to determine if the objects are similar
    threshold = 0.7
    if simil > threshold:
        print("The images are similar with a score of: ", simil)
    else:
        print("The images are not similar with a score of: ", simil)

hog_occurances(imgx,imgy)

def show_image(imgx, imgy):
    gray_image1 = cv2.cvtColor(imgx, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(imgy, cv2.COLOR_BGR2GRAY)
    gradx1 = cv2.Sobel(gray_image1, cv2.CV_32F, 1, 0, ksize=3)
    grady1 = cv2.Sobel(gray_image1, cv2.CV_32F, 0, 1, ksize=3)
    # Calculate the gradient magnitude and direction
    gradient_magnitude1 = np.sqrt(np.square(gradx1) + np.square(grady1))

    gradx2 = cv2.Sobel(gray_image2, cv2.CV_32F, 1, 0, ksize=3)
    grady2 = cv2.Sobel(gray_image2, cv2.CV_32F, 0, 1, ksize=3)
    # Calculate the gradient magnitude and direction
    gradient_magnitude2 = np.sqrt(np.square(gradx2) + np.square(grady2))

    cv2.imshow('image',imgx)   
    cv2.imshow('image1',imgy) 
    cv2.imshow("Gradient Image1", gradient_magnitude1)
    cv2.imshow("Gradient Image2", gradient_magnitude2)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    
    
    
show_image(imgx,imgy)

