
var points=[]
class NameForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {text: '',summary:'',points:''};

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }
  
  handleChange(event) {
    this.setState({text: event.target.value,summary:'',points:''});
    points=[]
  }

  handleSubmit(event) {
   
    var txt="{\"text\":\""+ this.state.text +"\"}";
      
    var xhr = new XMLHttpRequest();
    var z = this;
      
    xhr.open('post', '/api/mpg', true);
      
    xhr.onload = function () {
      var msg = JSON.parse(this.response)
      console.log("**" + msg.summary)
      
      for(var i=0;i<msg.npts;i++)
      {
        var str="msg.point";
        var temp=str.concat(String(i+1))
        points[i]=eval(temp)
      }
      console.log(points)
      z.setState({summary:`${msg.summary}`,points:`${points}`})
    };

    xhr.send(txt);
    console.log(z.state)
  
    event.preventDefault();

  }

  render() {
    return (
      <form onSubmit={this.handleSubmit} className="form" >
        <textarea rows="10" cols="100" className="textInput" placeholder="Copy and paste a news article here..." value={this.state.text} onChange={this.handleChange} />  
        <br/>
        <input type="submit" value="Generate Summary" className="btn btn-outline-dark" id="submitButton"/>
        <h3 id="head">{this.state.summary}</h3>
        <ul id="points">
          {points.map(points => <li>{points}</li>)}
        </ul>
      </form>
    );
  }
}

ReactDOM.render(
  <NameForm />,
  document.getElementById('root')
);