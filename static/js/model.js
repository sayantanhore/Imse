"use strict";

var IMG_GAL = IMG_GAL || {};

IMG_GAL.model = {
    
    image: function(imgObj){
        
        return ({
            index: imgObj.index,
            //path: 'static/images/im' + (parseInt(imgObj.index) + 1) + '.jpg',
            path: '/media/im' + (parseInt(imgObj.index) + 1) + '.jpg',
            rendered: false,
            feedback: 0.0
        })
    }
};
