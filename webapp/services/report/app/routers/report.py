from fastapi import APIRouter, HTTPException
from webapp.services.report.app.schemas.requests import GenerateReportRequest
from webapp.services.report.app.schemas.responses import (
    GenerateReportResponse
)
from webapp.services.report.app.utils.report_generator import (
    generate_report_data,
)
router = APIRouter()


@router.post("/generate_report", response_model=GenerateReportResponse)
async def generate_report(payload: GenerateReportRequest):
    """
    Generate a report by aggregating smells across multiple projects.
    """
    try:
        report_data = generate_report_data(payload.projects)
        return GenerateReportResponse(report_data=report_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating report: {str(e)}"
        )
