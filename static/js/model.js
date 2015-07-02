"use strict";

var Image = function(imageIndex, parent){
    this.imageIndex = imageIndex;
    this.parent = parent;
    this.imagePath = "../static/images/im" + imageIndex+ ".jpg";
    this.image = null;
    this.placed = false;
    this.feedback = 0.0;
    this.feedbackBox = null;
    
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
    //this.feedback = 0.0;
    this.feedbackBox.html(this.feedback);
    
    this.image.remove();
    this.image = $("<img/>").attr("src", this.imagePath);
    this.image.on("load", loadImage);
    this.parent.append(this.image);
    
    
};

