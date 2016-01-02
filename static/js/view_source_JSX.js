"use strict";


var MenuItem = React.createClass({
    mouseOverHandler: function(){
        $(ReactDOM.findDOMNode(this)).css('cursor', 'pointer');
        $(ReactDOM.findDOMNode(this)).css('text-decoration', 'underline');
    },
    
    mouseOutHandler: function(){
        $(ReactDOM.findDOMNode(this)).css('cursor', 'default');
        $(ReactDOM.findDOMNode(this)).css('text-decoration', 'none');
    },
    
    mouseClickHandler: function(){
	if(this.props.name === "Done"){
		IMG_GAL.controller.finished(
			{
				loc: IMG_GAL.globals.getLoc(),
				deRenderGallery: deRenderGallery
			}
		);
		alert("You're done, thank you!");
		return;
	}
        var imageObjs = IMG_GAL.globals.getImages();
        var feedbacked_image_indices = imageObjs.map(function(image){
            console.log("Index :: " + image.index);
	    return image.index
        });
	var feedback = imageObjs.map(function(image){
	    return image.feedback
	});
        console.log(feedbacked_image_indices, feedback)
	IMG_GAL.controller.predict(
            {
                loc: IMG_GAL.globals.getLoc(),
                numberOfImagesPerIteration: IMG_GAL.globals.getNumberOfImagesPerIteration(),
		feedbacked_image_indices: feedbacked_image_indices,
		feedback: feedback,
                renderGallery: renderGallery,
                deRenderGallery: deRenderGallery
            }
        );
    },
    
    render: function(){
        return (
            <span onMouseOver = {this.mouseOverHandler} onMouseOut = {this.mouseOutHandler} onClick = {this.mouseClickHandler}>{this.props.name}</span>
        );
    }
});

var MenuItems = React.createClass({
    
    render: function(){
        return (
            <div>{this.props.children}</div>
        );
    }
});

ReactDOM.render(
    <MenuItems>
        <MenuItem name = 'Next'></MenuItem>
        <MenuItem name = 'Done'></MenuItem>
    </MenuItems>,
    document.getElementById('menubar')
);


// *********************************************************************************************


var Gallery = React.createClass({
    getInitialState: function(){
        return {
            noOfImages: IMG_GAL.globals.getNumberOfImagesPerIteration(),
            images: IMG_GAL.globals.getImages(),
            howManyRendered: 0
        }
    },
    
    componentDidMount: function(){
        var elem = $(ReactDOM.findDOMNode(this));
        elem.css("height", IMG_GAL.util.attachUnit(IMG_GAL.globals.getScreenHeight() - IMG_GAL.util.getPxForEm(2.5)))
    },
    
    adjustLastRow: function(_$images, totalWidth, margin, imagesInCurrRow, lastRow){
        var firstIndex  = imagesInCurrRow[0];
        var lastIndex = imagesInCurrRow[imagesInCurrRow.length - 1];
        
        var heightToSetForAll = IMG_GAL.util.removeUnit(_$images.eq(lastIndex).css('height'));
        
        //alert("Change height of previous images")
        var occupiedWidth = IMG_GAL.util.removeUnit(_$images.eq(lastIndex).parent().css('width'));
        //var occupiedWidth = 0;
        _$images.slice(firstIndex, lastIndex).each(function(){
            $(this).css('height', IMG_GAL.util.attachUnit(heightToSetForAll));
            $(this).parent().css('height', IMG_GAL.util.attachUnit(heightToSetForAll));
            occupiedWidth += IMG_GAL.util.removeUnit($(this).parent().css('width'));
        });
        
        
        
        
        var spaceAvailable = totalWidth - occupiedWidth - (imagesInCurrRow.length - 1) * margin;
        console.log("Space empty :: " + spaceAvailable)
        heightToSetForAll += parseFloat(spaceAvailable * heightToSetForAll) / occupiedWidth;
        
        
        _$images.eq(lastIndex).parent().css('margin-right', '0em');
        
        _$images.slice(firstIndex, lastIndex + 1).each(function(index){
            var w = IMG_GAL.util.removeUnit($(this).css('width'));
            var h = IMG_GAL.util.removeUnit($(this).css('height'));
            $(this).css('height', IMG_GAL.util.attachUnit(heightToSetForAll));
            $(this).parent().css('height', IMG_GAL.util.attachUnit(heightToSetForAll));
            
            $(this).css('width', IMG_GAL.util.attachUnit(parseFloat(heightToSetForAll * w) / h));
            $(this).parent().css('width', IMG_GAL.util.attachUnit(parseFloat(heightToSetForAll * w) / h));
            
        });
        
    },
    
    adjustImage: function(){
        var totalWidth = IMG_GAL.util.removeUnit($('#gallery').css('width'));
        var availableHeight = IMG_GAL.util.removeUnit($('#gallery').css('height'));
        var rowHeightDivideFactor = IMG_GAL.globals.getRowHeightDivideFactor();
        var availableWidth = totalWidth;
        var margin = IMG_GAL.util.removeUnit($('#gallery .image-box').css('margin-right'));
        console.log(availableWidth);
        console.log(availableHeight);
        console.log(rowHeightDivideFactor);
        console.log(margin);
        
        var that = this;
        var imagesInCurrRow = [];
        
        var _$images = $('#gallery div img');
        _$images.hide();
        
        _$images.each(function(index){
            $(this).show();
            console.log("---------------------------")
            console.log(index)
            console.log(that.state.images[index].index)
            var heightToSet = IMG_GAL.util.attachUnit(parseFloat(availableHeight) / rowHeightDivideFactor);
            console.log("Height set")
            $(this).css('height', heightToSet);
            $(this).parent().css('height', heightToSet);
            
            var currHeight = IMG_GAL.util.removeUnit($(this).css('height'))
            var currWidth = IMG_GAL.util.removeUnit($(this).css('width'));
            console.log("Image width :: " + $(this).css('width'));
            
            if (availableWidth >= (currWidth + margin)){
                //alert("Space available")
                availableWidth -= (currWidth + margin);
                imagesInCurrRow.push(index);
            }
            else if(availableWidth >= (Math.ceil(currWidth * 0.75) + margin)){
                //alert("Changing width of last image")
                var widthToSet = Math.ceil(currWidth * 0.75);
                availableWidth -= (widthToSet + margin);
                
                $(this).css('width', IMG_GAL.util.attachUnit(widthToSet));
                $(this).parent().css('width', IMG_GAL.util.attachUnit(widthToSet));
                $(this).css('height', IMG_GAL.util.attachUnit(parseFloat(widthToSet * currHeight) / currWidth));
                $(this).parent().css('height', IMG_GAL.util.attachUnit(parseFloat(widthToSet * currHeight) / currWidth));
                imagesInCurrRow.push(index);
                if (that.state.noOfImages === (index +1)){
                    //alert("ebar")
                    that.adjustLastRow(_$images, totalWidth, margin, imagesInCurrRow, false);
                }
            }
            else{
                that.adjustLastRow(_$images, totalWidth, margin, imagesInCurrRow, false);
                
                availableWidth = totalWidth;
                availableWidth -= (currWidth + margin)
                
                imagesInCurrRow = [];
                imagesInCurrRow.push(index);
            }
        });
    },
    
    updateHowManyRendered: function(){
        var howManyRendered = this.state.howManyRendered;
        howManyRendered += 1
        this.setState({
            howManyRendered: howManyRendered
        });
        //console.log("Rendered :: " + this.state.howManyRendered);
        if (howManyRendered === this.state.noOfImages){
            console.log("calling")
            this.adjustImage();
        }
    },
    
    shouldComponentUpdate: function(){
        return false;
    },
    
    render: function(){
        return (
            <ImageBoxList images = {this.state.images} onImageLoad = {this.updateHowManyRendered}></ImageBoxList>
        );
    }
});

var ImageBoxList = React.createClass({
    
    render: function(){
        var that = this;
        var imageBoxList = this.props.images.map(function(imgObj){
            return (
                <ImageBox key = {imgObj.index} path = {imgObj.path} onImageLoad = {that.props.onImageLoad} imageObj = {imgObj}></ImageBox>
            );
        });
        return (
            <div id = 'gallery'>
                {imageBoxList}
            </div>
        );
    }
});

var ImageBox = React.createClass({
    
    getInitialState: function(){
        return {
            feedback: 0.0,
            clicked: false
        }
    },
    
    shouldComponentUpdate: function(){
        return false;
    },
    
    mouseMoveHandler: function(event){
        var _$imageBox = $(ReactDOM.findDOMNode(this));
        var _$feedbackBox = _$imageBox.find('.feedback-box');
        
        var mouse_x = event.pageX - _$imageBox.offset().left;
        var feedback = Math.abs((parseFloat(mouse_x) / IMG_GAL.util.removeUnit(_$imageBox.css('width')))).toFixed(1);
        _$feedbackBox.html(feedback);
        
        this.colourPicker(_$feedbackBox);
    },
    
    colourPicker: function(_$feedbackBox){
        var feedback = parseFloat(_$feedbackBox.html());
        if (feedback > 0.7){
            _$feedbackBox.css('background-color', 'green');
        }
        else if(feedback > 0.4 && feedback <= 0.7){
            _$feedbackBox.css('background-color', '#B18904');
        }
        else if(feedback <= 0.4){
            _$feedbackBox.css('background-color', '#DF3A01');
        }
    },
    
    mouseOverHandler: function(event){
        var _$imageBox = $(ReactDOM.findDOMNode(this));
        var _$feedbackBox = _$imageBox.find('.feedback-box');
        _$feedbackBox.html(this.state.feedback);
        _$feedbackBox.css('opacity', '0.8');
        
    },
    
    mouseOutHandler: function(event){
        var _$imageBox = $(ReactDOM.findDOMNode(this));
        var _$feedbackBox = _$imageBox.find('.feedback-box');
        if(! this.state.clicked){
            _$feedbackBox.css('opacity', '0.0');
        }
        else{
            _$feedbackBox.html(this.state.feedback);
        }
        this.colourPicker(_$feedbackBox);
        this.props.imageObj.feedback = parseFloat(this.state.feedback);
        console.log("Saved feedback :: " + this.props.imageObj.feedback)
    },
    
    mouseClickHandler: function(event){
        var _$imageBox = $(ReactDOM.findDOMNode(this));
        var _$feedbackBox = _$imageBox.find('.feedback-box');
        if (_$feedbackBox.html() !== '0.0'){
            this.setState({
                feedback: parseFloat(_$feedbackBox.html()),
                clicked: true
            });
        }
        else{
            this.setState({
                feedback: 0.0,
                clicked: false
            });
        }
    },
    
    componentDidMount: function(){
        
    },
    render: function(){
        return (
            <div className = 'image-box' onMouseMove = {this.mouseMoveHandler} onMouseOver = {this.mouseOverHandler} onMouseOut = {this.mouseOutHandler} onClick = {this.mouseClickHandler}>
                <FeedbackBox></FeedbackBox>
                <Image path = {this.props.path} onImageLoad = {this.props.onImageLoad}></Image>
            </div>
        );
        
    }
});

var FeedbackBox = React.createClass({
    componentDidMount: function(){
    
    },
    render: function(){
        return (
            <div className = 'feedback-box'>0.0</div>
        );
    }
});

var Image = React.createClass({
    onLoad: function(){
        console.log("Loading");
        this.props.onImageLoad();
    },
    render: function(){
        return (
            <img src = {this.props.path} onLoad = {this.onLoad}></img>
        );
    }
});

var renderGallery = function(){
    ReactDOM.render(
        <Gallery>
        </Gallery>,
        document.getElementById('container')
    );
};

var deRenderGallery = function(){
    ReactDOM.unmountComponentAtNode(document.getElementById('container'));
}

IMG_GAL.controller.firstround(
    {
        loc: IMG_GAL.globals.getLoc(),
        numberOfImagesPerIteration: IMG_GAL.globals.getNumberOfImagesPerIteration(),
        renderGallery: renderGallery
    }
);
