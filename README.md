#Traffic Sign Classifier

I firstly converted the 3 channel color image into a single channel image. I converted it into a luminance image. TO do that I multiplied by the coefficients suggested by the ITU-R recommendation. Luminance image is supposedly the way our human visual system perceives it. (It works well in our brain, So I'm guessing it should work well in this one too. xD). But, more importantly I did this so that there is less or no weightage on the type of color and the model is derived to look for the correct shapes and patterns and not color. I also applied a normalization step on the training and test data so that we have a uniform data representation.