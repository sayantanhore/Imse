// Creates an inset of an image

function createImageBox(acceptButton){

    return {
    
        imageBox: function(){
            console.log("Inside Image Box");
            var image = $("<img src='' alt='...'></img>");
            var link = $("<a href='#' class='thumbnail'></a>");

            var column = $("<div class = 'col-md-3'></div>");
            
            link.append(image);
            if(acceptButton === true){
                var header = $("<div class = 'image_header'></div>");
                header.append("<span class = 'glyphicon glyphicon-ok circular-border'></span>");
                column.append(header);
            }
            else{
                
                column.append(attachSlider());
            }

            column.append(link);
        
            return column
            
        }
    
    };
    
};

function attachSlider(){
    var input_slider = $("<input type='text' class='' value='' data-slider-min='0' data-slider-max='10' data-slider-step='1' data-slider-value='5' data-slider-orientation='horizontal' style='margin-top: 50px' data-slider-handle='round' />");
    var slider_container = $("<div class = 'slider-header'></div>");
    slider_container.append(input_slider);
    return slider_container;
}