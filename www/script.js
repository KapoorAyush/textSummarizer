

class NameForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {text: '',summary:''};

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
      z.setState({summary:`${msg.summary}`})
      
      } ;

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
      </form>
    );
  }
}

ReactDOM.render(
  <NameForm />,
  document.getElementById('root')
);