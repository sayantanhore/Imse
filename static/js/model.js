"use strict";

var Image = function(imageIndex){
    this.imageIndex = imageIndex;
    this.imagePath = "../static/images/im" + imageIndex+ ".jpg";
    this.image = undefined;
    this.placed = false;
    
    this.loc = {
        top: -999,
        left: -999
    };
    
    this.dim = {
        height: 0,
        width: 0
    }
};

Image.prototype.changeImage = function(imageIndex){
    this.imageIndex = imageIndex;
    this.imagePath = "../static/images/im" + imageIndex+ ".jpg";
    
    $("#container").empty();
    
    var startIndex = 0;
    availableWidth = $("#container").width() - 2 * marginWidth;
    setImageInPlace(containerHeight, containerWidth, availableWidth, startIndex);
};