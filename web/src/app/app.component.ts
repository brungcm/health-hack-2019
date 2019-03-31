import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval } from 'rxjs';
import { switchMap } from 'rxjs/operators';

interface IRiskData {
    riskPosition: number;
    faceWaterSeconds: number;
    riskPanic: number;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
    baseUrl = 'http://localhost:5000/status';
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

    weight = {
        position: 1,
        submerged: 4.5,
        panic: 2
    };

    totalRiskVal = 0;
    totalRiskTxt = '';

    constructor(private httpClient: HttpClient) {
    }

    public ngOnInit() {
        this.requestForever();
    }

    private requestForever() {
        const result = interval(1000).pipe(
            switchMap(() => this.httpClient.get(this.baseUrl))
        );

        result.subscribe(
            (data: IRiskData) => {
                this.chartDatas[0].gaugeValue = data.riskPosition;
                this.chartDatas[1].gaugeValue = this.calcFaceWaterPercent(data.faceWaterSeconds);
                this.chartDatas[2].gaugeValue = data.riskPanic;

                this.totalRiskVal = this.calcRisk(data);
                this.totalRiskTxt = this.getRiskTxt(this.totalRiskVal);
            },
            (error: Error) => {
                console.error(error);
            }
        );
    }

    private calcFaceWaterPercent(faceWaterSeconds: number): number {
        const value = (faceWaterSeconds / 30) * 100;
        return value > 100 ? 100 : Math.floor(value);
    }

    private calcRisk(data: IRiskData): number {
        const submergedValue = this.calcFaceWaterPercent(data.faceWaterSeconds);
        const weightTotal = (this.weight.position + this.weight.submerged + this.weight.panic);
        const result = ((data.riskPosition * this.weight.position) + (submergedValue * this.weight.submerged) + (data.riskPanic * this.weight.panic))/weightTotal;

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
