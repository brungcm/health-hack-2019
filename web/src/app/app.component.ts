import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval } from 'rxjs';
import { switchMap } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
    baseUrl = 'http://localhost:5002/';
    totalRisk = 0;
    chartDatas: any[] = [
        {
            gaugeType: 'semi',
            gaugeValue: 0,
            gaugeLabel: 'Position'
        },
        {
            gaugeType: 'semi',
            gaugeValue: 0,
            gaugeLabel: 'Submerged Face'
        },
        {
            gaugeType: 'semi',
            gaugeValue: 0,
            gaugeLabel: 'Panic'
        }
    ];

    thresholdConfig = {
        0: { color: '#7fffd4' },
        30: { color: '#FF8C00' },
        70: { color: '#ed0215' }
    };

    constructor(private httpClient: HttpClient) {
    }

    public ngOnInit() {
        this.requestForever();
    }

    private requestForever() {
        const result = interval(3000).pipe(
            switchMap(() => this.httpClient.get(this.baseUrl))
        );

        result.subscribe(
            (data: any) => {
                this.chartDatas[0].gaugeValue = data.riskPosition;
                this.chartDatas[1].gaugeValue = data.submergedFaceSeconds;
                this.chartDatas[2].gaugeValue = data.riskPanic;

                this.totalRisk = Math.floor((data.riskPosition + data.submergedFaceSeconds + data.riskPanic) / 3);
            },
            (error: Error) => {
                console.error(error);
            }
        );
    }

}
