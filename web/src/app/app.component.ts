import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval } from 'rxjs';
import { switchMap } from 'rxjs/operators';

interface IRiskData {
    riskPosition: number;
    submergedFaceSeconds: number;
    riskPanic: number;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
    baseUrl = 'http://localhost:5002/';
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

    riskLevel = {
        low: 'Low',
        medium: 'Medium',
        high: 'High'
    };

    totalRiskVal = 0;
    totalRiskTxt = '';

    constructor(private httpClient: HttpClient) {
    }

    public ngOnInit() {
        this.requestForever();
    }

    private requestForever() {
        const result = interval(10000).pipe(
            switchMap(() => this.httpClient.get(this.baseUrl))
        );

        result.subscribe(
            (data: IRiskData) => {
                this.chartDatas[0].gaugeValue = data.riskPosition;
                this.chartDatas[1].gaugeValue = data.submergedFaceSeconds;
                this.chartDatas[2].gaugeValue = data.riskPanic;

                this.totalRiskVal = this.calcRisk(data);
                this.totalRiskTxt = this.getRiskTxt(this.totalRiskVal);
            },
            (error: Error) => {
                console.error(error);
            }
        );
    }

    private calcRisk(data: IRiskData): number {
        const weight = {
            position: 1,
            submerged: 4.5,
            panic: 2
        };

        const weightTotal = (weight.position + weight.submerged + weight.panic);
        const result = ((data.riskPosition * weight.position) + (data.submergedFaceSeconds * weight.submerged) + (data.riskPanic * weight.panic))/weightTotal;

        return Math.floor(result);

    }

    private getRiskTxt(result: number): string {
        if (result <= 33) {
            return this.riskLevel.low;
        }

        if (result > 33 && result < 66) {
            return this.riskLevel.medium;
        }

        return this.riskLevel.high;
    }

}
