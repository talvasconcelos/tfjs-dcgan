import { Component } from 'preact'
import {draw} from '../ui.js'

class Canvas extends Component {

    state = {
        image: this.props.data
    }

    componentDidMount = () => {
        const image = this.state.image
        const canvas = this.canvas
        canvas.width = 28
        canvas.height = 28
        const ctx = canvas.getContext("2d")
        draw(image, canvas)
      }

    render({}, {image}) {
        return(
            <div>
                <canvas ref={canvas => this.canvas = canvas} />
            </div>    
        )
      }
}

export default Canvas