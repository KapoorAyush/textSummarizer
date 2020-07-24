

class NameForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {text: '',summary:'',points:''};

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({text: event.target.value,summary:''});
  }

  handleSubmit(event) {
   
    var txt="{\"text\":\""+ this.state.text +"\"}";
      
    var xhr = new XMLHttpRequest();
    var z = this;
      
    xhr.open('post', '/api/mpg', true);
      
    xhr.onload = function () {
      var msg = JSON.parse(this.response)
      console.log("**" + msg.summary)
      var points=''
      for(var i=1;i<=msg.npts;i++)
      {
        var temp="msg.point".concat(String(i))
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
      <form onSubmit={this.handleSubmit}>
        <label>
          Name:
          <input type="text" value={this.state.text} onChange={this.handleChange} />
        </label>
        <input type="submit" value="Submit" />
        <span>{this.state.summary}</span>
        <ul>
          <li>{this.state.points}</li>
        </ul>
      </form>
    );
  }
}

ReactDOM.render(
  <NameForm />,
  document.getElementById('root')
);