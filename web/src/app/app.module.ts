import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';

import { AppComponent } from './app.component';
import { NgxGaugeModule } from 'ngx-gauge';


@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    NgxGaugeModule

  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
