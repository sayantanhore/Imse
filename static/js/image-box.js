// ImageBox component

// Author Sayantan Hore
// Created on 19.08.2014

// --------------------------------

function ImageBox(){
    this.outerContainer = null;
    this.imageBoxArray = [];
}
ImageBox.prototype.createOuterContainer = function(){
    this.outerContainer = $("<div></div>")
    this.outerContainer.addClass("outer-image-container");
    $("body").append(this.outerContainer);
    
}
ImageBox.prototype.createImageBox = function(){
    if (this.createOuterContainer === null){
        this.createOuterContainer();
    }
    var imageBox = $("<div></div>");
    imageBox.addClass("thumbnail");
    var closeButton = $("<div>x</div>");
    closeButton.addClass("btn-close");
    
    //imageBox.append(closeButton);
    this.imageBoxArray.push(imageBox);
    return imageBox;
}

ImageBox.prototype.loadImage = function(imPath, imageBox){
    var img = $("<img></img>")
    img.attr("src", imPath);
    imageBox.append(img);

    img.load(function(){
        //console.log($(this).height());
        imgWidth = $(this).width();
        $(this).parent().height(rowHeight);
        $(this).height(rowHeight);
        //console.log($(this).width());
        $(this).parent().width($(this).width());
    });
}