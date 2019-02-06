import { Component } from 'preact'
import {draw} from '../ui.js'

class Canvas extends Component {

    updateCanvas = () => {
        const image = this.props.data
        const canvas = this.canvas
        canvas.width = 28
        canvas.height = 28
        const ctx = canvas.getContext("2d")
        draw(image, canvas)
    }

    componentDidUpdate = (prevProps) => {
        if (this.props.data !== prevProps.data) {
            this.updateCanvas()
          }
    }

    componentDidMount = () => {
        this.updateCanvas()
        // const image = this.state.image
        // const canvas = this.canvas
        // canvas.width = 28
        // canvas.height = 28
        // const ctx = canvas.getContext("2d")
        // draw(image, canvas)
      }

    render() {
        return(
            <div>
                <canvas ref={canvas => this.canvas = canvas} />
            </div>    
        )
      }
}

export default Canvas