import {drawCanvas, getImages, showTestResults} from '../ui.js'
import Canvas from './canvas'

const Samples = ({examples}) => {
    
    return (
        <div>
            <h3>Samples</h3>
            <div id="testSamples">
                {examples && getImages(examples).map((img, i) => <Canvas key={i} data={img}/>)}
                {/* {showTestResults(examples)} */}
            </div>
        </div>
    )
}

export default Samples