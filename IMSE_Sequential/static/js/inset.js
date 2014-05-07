// Creates an inset of an image

function createInset(){

    return {
    
        imageBox: function(){
            
            var imageBox = $("<div class = 'inset'></div>");
            var header = $("<div></div>");
            header.append("<span class = 'glyphicon glyphicon-ok circular-border'></span>")
            header.append("<span class = 'glyphicon glyphicon-remove circular-border'></span>");
            header.addClass("thumbnail");
            imageBox.append(header);
            imageBox.append($("<img src='' alt='...'>"));
            imageBox_width = $("#id_image_container .panel-body .row .col-md-3 a").css("width").replace("px", "") * 60 / 100;
            imageBox_top = $("#id_image_container .panel-body .row .col-md-3").css("top");
            imageBox_left = $("#id_image_container .panel-body .row .col-md-3").css("left");
            imageBox.css("width", imageBox_width + "px");
            imageBox.css("height", imageBox_width + "px");
            imageBox.css("top", imageBox_top + "px");
            imageBox.css("left", imageBox_left + "px");
            imageBox.addClass("thumbnail");
            
            return imageBox;
            
        }
    
    };
    
};