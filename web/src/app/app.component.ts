import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'app';
  allData: JSON;

  constructor(private httpClient: HttpClient) {
  }

  ngOnInit() {
  }

  findAll() {
    this.httpClient.get('http://127.0.0.1:5002/').subscribe(data => {
      this.allData = data as JSON;
      console.log(this.allData);
    })
  }
}
